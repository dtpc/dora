from gpflow.gpr import GPR
from gpflow.mean_functions import Zero
from gpflow.param import DataHolder
from gpflow.param import AutoFlow
from gpflow import session
from gpflow._settings import settings

import tensorflow as tf
import numpy as np
import sys


class GPRCached(GPR):
    """ GPflow.gpr.GPR class that stores Cholesky decomposition for efficiency
        and performs single row Cholesky update and downdate computations.

        Caching based on https://github.com/GPflow/GPflow/issues/333
    """

    def __init__(self, x, y, kern, mean_function=Zero(), name='name'):
        """Initialize GP and Cholesky decomposition."""
        GPR.__init__(self, x, y, kern=kern, mean_function=mean_function,
                     name=name)

        # Create new dataholders for the cached data
        self.cholesky = DataHolder(np.empty((0, 0), dtype=np.float64),
                                   on_shape_change='pass')
        self.alpha = DataHolder(np.empty((0, 0), dtype=np.float64),
                                on_shape_change='pass')
        self.update_cache()

    def __setattr__(self, key, value):
        """ Disallow setting `X` and `Y` directly, so that cached
            computations remain in sync."""
        if key in ('X', 'Y') and hasattr(self, key):
            raise ValueError('Changes to X and Y should be made through calls '
                             'to `set_data_points(X, Y)`')

        GPR.__setattr__(self, key, value)

    def set_parameter_dict(self, d):
        """ Update cache when parameters are reset. """
        GPR.set_parameter_dict(self, d)
        self.update_cache()

    def set_state(self, x):
        """ Update cache when parameters are reset.

            `set_state` is called during `optimize`.
        """
        GPR.set_state(self, x)
        self.update_cache()

    def _cholesky(self, X, Y):

        kernel = (self.kern.K(X)
                  + tf.eye(tf.shape(X)[0], dtype=np.float64)
                  * self.likelihood.variance)

        cholesky = tf.cholesky(kernel, name='gp_cholesky')
        target = Y - self.mean_function(X)
        alpha = tf.matrix_triangular_solve(cholesky, target, name='gp_alpha')
        return cholesky, alpha

    @AutoFlow()
    def _compute_cache(self):
        """Compute cache."""
        return self._cholesky(self.X, self.Y)

    def update_cache(self):
        """Update the cache after adding data points."""
        self.cholesky, self.alpha = self._compute_cache()

    @AutoFlow((tf.float64, [None, None]))
    def _cholesky_update(self, X):
        """ Perform incremental update of Cholesky decomposition by adding
            data point(s) `X`.
        """
        kxn = self.kern.K(self.X, X)
        knn = (self.kern.K(X, X)
              + tf.eye(tf.shape(X)[0], dtype=np.float64)
              * self.likelihood.variance)

        L = self.cholesky
        c = tf.matrix_triangular_solve(L, kxn, lower=True)
        d = tf.cholesky(knn - tf.matmul(tf.transpose(c), c))

        cholesky = tf.concat([
            tf.concat([L, tf.zeros(tf.shape(c), dtype=tf.float64)], axis=1),
            tf.concat([tf.transpose(c), d], axis=1)
        ], axis=0, name='gp_cholesky_update')

        return cholesky

    @AutoFlow((tf.int32,), (tf.int32,))
    def _cholesky_downdate(self, i, n=1):
        """ Perform downdate of Cholesky decomposition by removing n
            consecutive points data point at index `i`.
        """
        L = self.cholesky
        m = tf.shape(L)[0] - (i + n)

        Sa = tf.slice(L, begin=[i+n, i], size=[m, n])
        Sb = tf.slice(L, begin=[i+n, i+n], size=[m, m])
        R = tf.cholesky(tf.add(
                tf.matmul(Sa, tf.transpose(Sa)),
                tf.matmul(Sb, tf.transpose(Sb))
        ))

        left = tf.concat([
            tf.slice(L, begin=[0, 0], size=[i, i]),
            tf.slice(L, begin=[i+n, 0], size=[m, i]),
            ], axis=0)

        right = tf.concat([tf.zeros([i, m], dtype=tf.float64), R], axis=0)
        cholesky = tf.concat([left, right], axis=1, name='gp_cholesky_downdate')

        return cholesky

    @AutoFlow()
    def _alpha_update(self):
        """ Compute alpha (use after `self.cholesky` has been updated). """
        target = self.Y - self.mean_function(self.X)
        alpha = tf.matrix_triangular_solve(self.cholesky, target,
                                           name='gp_alpha_update')
        return alpha

    def set_data_points(self, X, Y):
        """ Reset the data points to arrays `X` and `Y`. Update cache. """
        assert X.shape[0] == Y.shape[0]
        GPR.__setattr__(self, 'X', X)
        GPR.__setattr__(self, 'Y', Y)
        self.update_cache()

    def add_data_points(self, X, Y):
        """ Add data point(s) (`X`, `Y`) to GP and perform update of Cholesky
            decomposition.
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        assert X.shape[0] == Y.shape[0]

        self.cholesky = self._cholesky_update(X)
        self.X.set_data(np.append(self.X.value, X, axis=0))
        self.Y.set_data(np.append(self.Y.value, Y, axis=0))
        self.alpha = self._alpha_update()

    def remove_data_points(self, indexes):
        """ Remove points at `indexes` from both X and Y, and downdate Cholesky
            decomposition.

            Currently not faster than full decomposition calculation.
        """
        if not hasattr(indexes, '__iter__'):
            indexes = np.array([indexes])

        if not isinstance(indexes, np.ndarray):
            indexes = np.array(indexes)

        indexes = np.unique(indexes)

        if indexes[0] < 0 or indexes[-1] > self.X.shape[0]:
            raise IndexError('Indexes out of range {}.'.format(indexes))

        index_grps = np.split(indexes, np.where(np.diff(indexes) != 1)[0] + 1)
        index_blks = [(g[0], len(g)) for g in reversed(index_grps)]

        # should be able to rewrite `_cholesky_downdate` to do these all at
        # once efficiently
        for i, n in index_blks:
            self.cholesky = self._cholesky_downdate(i, n)

        self.X.set_data(np.delete(self.X.value, indexes, axis=0))
        self.Y.set_data(np.delete(self.Y.value, indexes, axis=0))
        self.alpha = self._alpha_update()

    def _predict(self, X, Xnew, Y, cholesky, alpha, full_cov=False):
        Kx = self.kern.K(X, Xnew)
        A = tf.matrix_triangular_solve(cholesky, Kx, lower=True)
        fmean = (tf.matmul(tf.transpose(A), alpha) + self.mean_function(Xnew))
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.stack([1, 1, tf.shape(Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(Y)[1]])
        return fmean, fvar

    def build_predict(self, Xnew, full_cov=False):
        """Predict mean and variance of the GP at locations in Xnew.

        Parameters
        ----------
        Xnew : ndarray
            The points at which to evaluate the function. One row for each
            data points.
        full_cov : bool
            if False returns only the diagonal of the covariance matrix

        Returns
        -------
        mean : ndarray
            The expected function values at the points.
        error_bounds : ndarray
            Diagonal of the covariance matrix (or full matrix).

        """
        fmean, fvar = self._predict(self.X, Xnew, self.Y, self.cholesky,
                                    self.alpha, full_cov=full_cov)
        return fmean, fvar

    def build_loo(self):

        def density_i(X, Xnew, Y, Ynew):
            cholesky, alpha = self._cholesky(X, Y)
            fmean, fvar = self._predict(X, Xnew, Y, cholesky, alpha,
                                        full_cov=False)
            logp_i = self.likelihood.predict_density(fmean, fvar, Ynew)
            return logp_i

        i_sum = (tf.constant(0), tf.constant(0.0))

        def body(i, sum):
            Xnew = tf.slice(self.X, [i, 0], [1, -1])
            Ynew = tf.slice(self.Y, [i, 0], [1, -1])

            js = tf.concat(tf.range(i), tf.range(i+1, tf.shape(self.X)[0]))
            X = tf.gather(self.X, js)
            Y = tf.gather(self.Y, js)

            logp_i = density_i(X, Xnew, Y, Ynew)

            return tf.add(i, 1), tf.add(sum, logp_i)

        def condition(i, sum):
            return tf.less(i, tf.subtract(tf.shape(self.X)[0], 1))

        sum_logp = tf.while_loop(condition, body, i_sum)[1]
        return sum_logp

    def build_loo_cv(self, X, Y):

        def density_i(i):
            Xi = tf.constant(X[np.newaxis, i, :])
            Yi = tf.constant(Y[np.newaxis, i, :])
            Xjs = tf.constant(np.delete(X, i, axis=0))
            Yjs = tf.constant(np.delete(Y, i, axis=0))
            cholesky, alpha = self._cholesky(Xjs, Yjs)
            fmean, fvar = self._predict(Xjs, Xi, Yjs, cholesky, alpha,
                                        full_cov=False)
            logp_i = self.likelihood.predict_density(fmean, fvar, Yi)
            return logp_i

        logp_is = [density_i(i) for i in range(X.shape[0])]
        sum_logp = tf.reduce_sum(logp_is)

        return sum_logp

    def optimize(self, loo_cv=False, method='L-BFGS-B', tol=None, callback=None,
                 maxiter=1000, **kw):
        self._compile(loo_cv=loo_cv)
        res = super().optimize(method=method, tol=tol, callback=callback,
                                    maxiter=maxiter, **kw)
        return res

    def _compile(self, optimizer=None, loo_cv=False):
        """
        compile the tensorflow function "self._objective"
        """
        self._graph = tf.Graph()
        self._session = session.get_session(graph=self._graph,
                                            output_file_name=settings.profiling.output_file_name + "_objective",
                                            output_directory=settings.profiling.output_directory,
                                            each_time=settings.profiling.each_time)
        with self._graph.as_default():
            self._free_vars = tf.Variable(self.get_free_state())

            self.make_tf_array(self._free_vars)

            if loo_cv:
                X = self.X.value
                Y = self.Y.value
                with self.tf_mode():
                    f = self.build_loo_cv(X, Y)
                    g, = tf.gradients(f, self._free_vars)
            else:
                with self.tf_mode():
                    f = self.build_likelihood() + self.build_prior()
                    g, = tf.gradients(f, self._free_vars)

            self._minusF = tf.negative(f, name='objective')
            self._minusG = tf.negative(g, name='grad_objective')

            # The optimiser needs to be part of the computational graph, and needs
            # to be initialised before tf.initialise_all_variables() is called.
            if optimizer is None:
                opt_step = None
            else:
                opt_step = optimizer.minimize(self._minusF,
                                              var_list=[self._free_vars])
            init = tf.global_variables_initializer()
        self._session.run(init)

        # build tensorflow functions for computing the likelihood
        if settings.verbosity.tf_compile_verb:
            print("compiling tensorflow function...")
        sys.stdout.flush()

        self._feed_dict_keys = self.get_feed_dict_keys()

        def obj(x):
            self.num_fevals += 1
            feed_dict = {self._free_vars: x}
            self.update_feed_dict(self._feed_dict_keys, feed_dict)
            f, g = self._session.run([self._minusF, self._minusG],
                                     feed_dict=feed_dict)
            return f.astype(np.float64), g.astype(np.float64)

        self._objective = obj
        if settings.verbosity.tf_compile_verb:
            print("done")
        sys.stdout.flush()
        self._needs_recompile = False

        return opt_step
