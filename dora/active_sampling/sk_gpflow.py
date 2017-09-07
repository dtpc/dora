import gpflow as gp
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.pipeline import if_delegate_has_method
from tqdm import tqdm
from .likelihoods import likelihood_compute
import numpy as np
from scipy.special import erfinv


class PrintCallback:
    def __init__(self, maxiter, updateiter=None):
        self.ii = 0
        self.updateiter = updateiter or int(maxiter / 100)
        self.pbar = tqdm(desc='GPflow iterations', total=maxiter)

    def __call__(self, k):
        if self.ii % self.updateiter == 0:
            self.pbar.update(self.updateiter)
        self.ii += 1

    def __del__(self):
        self.pbar.close()


def gaussian_quantiles(q):
    return np.sqrt(2)*erfinv(2*np.asarray(q) - 1)


class SkGPflowRegressor(BaseEstimator, RegressorMixin):
    """ sklearn-like wrapper around GPflow regressors. """

    def __init__(self, kern, optimize_args=None, **model_args):
        self.kern = kern

        default_args = {
            'mean_function': gp.mean_functions.Constant()
        }

        for k, v in default_args.items():
            if k not in model_args:
                model_args[k] = v

        # Add compute_* functions to likelihood model
        if 'likelihood' in model_args:
            model_args['likelihood'] = likelihood_compute(model_args['likelihood'])

        self.model_args = model_args
        self.name = 'name'

        self.optimize_args = optimize_args or dict(method='L-BFGS-B',
                                                   maxiter=10000)
        self.n_targets = None

    def _create_model(self, X, y):
        raise NotImplementedError()

    def fit(self, X, y, param_dict=None, optimize_args=None):
        """ Fit the GP model. Optimise the model if params is not provided. """
        if X.shape[1] != self.kern.input_dim:
            msg = 'Dimensions of `x` ({}) do not match kernel dimensions ({})'
            raise ValueError(msg.format(X.shape[1], self.kern.input_dim))

        y = np.atleast_2d(y)
        self.n_targets = y.shape[1]

        # create the GPflow model
        self._create_model(X, y)

        if param_dict is None:
            optimize_args = optimize_args or {}
            self.optimize(**optimize_args)
        else:
            self.model.set_parameter_dict(param_dict)

        return self

    def predict(self, X, return_std=False, return_cov=False):
        """ Return the GPs expected value and variance at the points in `x`. """
        check_is_fitted(self, 'model')
        y_exp, y_var = self.model.predict_y(X)
        res = y_exp

        if return_std:
            res = (res, np.sqrt(y_var))
        if return_cov:
            raise NotImplementedError

        return res

    def predict_quantiles(self, X, quantiles=None):
        """ Return the predicted value y at defined quantiles.

            output shape (n_samples, n_targets, n_quantiles)
        """
        q = quantiles or [0.025, 0.975]
        samps = gaussian_quantiles(q)
        mu, var = self.model.predict_f(X)
        latent_q = (mu[:, :, np.newaxis]
                    + np.sqrt(var[:, :, np.newaxis]) * samps[np.newaxis, :])
        qmu = self.model.likelihood.compute_conditional_mean(latent_q)
        qvar = self.model.likelihood.compute_conditional_variance(latent_q)
        q_exp = qmu + np.sqrt(qvar) * samps
        return q_exp

    def is_fitted(self):
        """ Return True if fit() has been called on this model instance. """
        fitted = True
        try:
            check_is_fitted(self, 'model')
        except NotFittedError:
            fitted = False
        return fitted

    def optimize(self, **kwargs):
        """ Optimise the GP hyperparams. """
        optimize_args = self.optimize_args.copy()
        optimize_args.update(kwargs)
        optimize_args['callback'] = PrintCallback(optimize_args['maxiter'])
        res = self.model.optimize(**optimize_args)
        return res

    def get_params1(self, deep=False):
        """ Get GP parameters in sklearn form. Keyword arg `deep` is ignored."""
        check_is_fitted(self, 'model')
        p_dict = self.model.get_parameter_dict()

        def sk_param_name(name):
            return name.split('.', 1)[1].replace('.', '__')

        params = {sk_param_name(k): v for k, v in p_dict.items()}
        return params

    def set_params1(self, **params):
        """ Set GP params using sklearn form. """
        check_is_fitted(self, 'model')
        p_dict = self.model.get_parameter_dict()

        for sk_name, param in params.items():
            name = '.'.join([self.name] + sk_name.split('__'))
            if name not in p_dict:
                msg = "GPflowRegressor '{}' has no parameter '{}'."
                raise KeyError(msg.format(self.name, sk_name))
            p_dict[name] = param

        self.model.set_parameter_dict(p_dict)

    def __repr__1(self):
        """ sklearn representation with GPflow model name. """
        class_name = self.__class__.__name__
        gp_class_name = self.gp_class.__module__ + "." + self.gp_class.__name__
        class_and_model = '{}[{}]'.format(class_name, gp_class_name)
        if self.is_fitted():
            s = super().__repr__()
            s.replace(class_name, class_and_model)
        else:
            s = "{}('Model not instantiated.')".format(class_and_model)

        return s


class SkGPR(SkGPflowRegressor):
    """ sklearn compatible wrapper for GPflow.gpr.GPR """

    def __init__(self, kern, optimize_args=None, **model_args):
        super().__init__(kern, optimize_args, **model_args)
        self.gp_class = gp.gpr.GPR

    def _create_model(self, X, y):
        self.model = self.gp_class(X, y, kern=self.kern, **self.model_args)
        self.model.likelihood = likelihood_compute(self.model.likelihood)

    def predict_quantiles(self, X, quantiles=None):
        """ Return the predicted value y at defined quantiles. """
        q = quantiles or [0.025, 0.975]
        samps = gaussian_quantiles(q)
        mu, var = self.model.predict_y(X)
        q_exp = (mu[:, :, np.newaxis]
                 + np.sqrt(var[:, :, np.newaxis]) * samps[np.newaxis, :])
        return q_exp


class SkSGPR(SkGPflowRegressor):
    """ sklearn compatible wrapper for GPflow.sgpr.SGPR """

    def __init__(self, kern, inducing, random_seed=None, optimize_args=None,
                 **model_args):
        super().__init__(kern, optimize_args, **model_args)
        self.gp_class = gp.sgpr.SGPR
        self.inducing = inducing
        self.random_seed = random_seed

    def _create_model(self, X, y):
        km = MiniBatchKMeans(n_clusters=self.inducing,
                             random_state=self.random_seed)
        Z = km.fit(X).cluster_centers_
        self.model = self.gp_class(X, y, kern=self.kern, Z=Z, **self.model_args)
        self.model.likelihood = likelihood_compute(self.model.likelihood)


class SkVGP(SkGPflowRegressor):
    """ sklearn compatible wrapper for GPflow.vgp.VGP """

    def __init__(self, kern, optimize_args=None, **model_args):
        super().__init__(kern, optimize_args, **model_args)
        self.gp_class = gp.vgp.VGP

    def _create_model(self, X, y):
        self.model = self.gp_class(X, y, kern=self.kern, **self.model_args)


class SkSVGP(SkGPflowRegressor):
    """ sklearn compatible wrapper for GPflow.svgp.SVGP """

    def __init__(self, kern, inducing, random_seed=None, optimize_args=None,
                 **model_args):
        super().__init__(kern, optimize_args, **model_args)
        self.gp_class = gp.svgp.SVGP
        self.inducing = inducing
        self.random_seed = random_seed

    def _create_model(self, X, y):
        km = MiniBatchKMeans(n_clusters=self.inducing,
                             random_state=self.random_seed)
        Z = km.fit(X).cluster_centers_
        self.model = self.gp_class(X, y, kern=self.kern, Z=Z,
                                   minibatch_size=100, **self.model_args)


class MultiTargetRegressor(BaseEstimator, RegressorMixin):
    """ Simple version of sklearn.multioutput.MultiOutputRegressor without
        parallel jobs to get around pickling problems with gpflow/tensorflow.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators_ = []

    def fit(self, X, y,):
        """ Create and fit a separate model for each output. """
        def create_fit(X, yi):
            est = clone(self.estimator)
            est.fit(X, yi)
            return est

        self.estimators_ = [create_fit(X, y[:, i, np.newaxis]) for i in range(
            y.shape[1])]

    def predict(self, X, return_std=False, return_cov=False):
        """ Run predict on each model and stack outputs. """
        check_is_fitted(self, 'estimators_')

        if return_std or return_cov:
            raise NotImplementedError

        res = np.hstack([e.predict(X) for e in self.estimators_])
        return res

    def predict_quantiles(self, X, quantiles=None):
        """ Run predict_quantiles on each model and stack outputs. """
        check_is_fitted(self, 'estimators_')

        res = np.hstack([e.predict_quantiles(X, quantiles=quantiles) for e in
                         self.estimators_])
        return res


class PipelineX(Pipeline):
    """ Like sklearn.pipeline.Pipeline but with `y_transform` and
        `predict_quantiles()`.
    """

    def __init__(self, steps, y_transform=None):
        super().__init__(steps)
        self.y_transform = y_transform

    def fit(self, X, y, **fit_model_kwargs):
        """ Fit y_transform and model. """
        if self.y_transform:
            self.y_transform.fit(y, None)
            y = self.y_transform.transform(y)
        res = super().fit(X, y, **fit_model_kwargs)
        return res

    def fit_y_transform(self, y, **kwargs):
        """ Fit y_transform only. """
        if self.y_transform:
            self.y_transform.fit(y, **kwargs)
        return self

    def fit_model(self, X, y, **kwargs):
        """ Fit model only. """
        if self.y_transform:
            y = self.y_transform.transform(y)
        res = super().fit(X, y, **kwargs)
        return res

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, y_transform=True, **final_est_kwargs):
        """ Predict model and apply inverse y transform on estimate. """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)

        res = self.steps[-1][-1].predict(Xt, **final_est_kwargs)

        if self.y_transform and y_transform:
            if isinstance(res, tuple):
                res = tuple(map(self.y_transform.inverse_transform, res))
            else:
                res = self.y_transform.inverse_transform(res)

        return res

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_quantiles(self, X, quantiles=None, y_transform=True):
        """ Apply transforms, and predict_quantiles of the final estimator
            returns: y_q [n_samples, n_quantiles]
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)

        y_q = self.steps[-1][-1].predict_quantiles(Xt, quantiles)

        if self.y_transform and y_transform:
            y_r = np.rollaxis(y_q, 2)
            y_q = np.stack(
                [self.y_transform.inverse_transform(y) for y in y_r],
                axis=2
            )

        return y_q

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None):

        if self.y_transform:
            self.y_transform.fit(y, None)
            y = self.y_transform.transform(y)

        s = super().score(X, y)
        return s


class CoregGP(SkGPflowRegressor):
    def __init__(self, kern):  # , likelihood):
        super().__init__(kern=kern)  # , likelihood=likelihood)
        self.gp_class = gp.gpr.GPR

    def _create_model(self, X, y):
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        # coregion kernel with random initial weights
        coreg = gp.kernels.Coregion(1, output_dim=self.output_dim, rank=1,
                                    active_dims=[self.input_dim])
        coreg.W = np.random.randn(self.output_dim, 1)

        kern = self.kern * coreg

        X_a = self._coreg_data_transform_x(X)
        y_a = self._coreg_data_transform_y(y)

        from IPython import embed; embed()

        self.model = self.gp_class(X_a, y_a, kern=kern, **self.model_args)

        self.model.likelihood = likelihood_compute(self.model.likelihood)

    def predict(self, X, return_std=False, return_cov=False):
        X_a = self._coreg_data_transform_x(X)
        res_a = super().predict(X_a, return_std=return_std,
                                return_cov=return_cov)

        if return_std or return_cov:
            res = tuple(map(self._coreg_data_inverse_transform_y, res_a))
        else:
            res = self._coreg_data_inverse_transform_y(res_a)
        return res

    def predict_quantiles(self, X, quantiles=None):
        X_a = self._coreg_data_transform_x(X)
        yq_a = super().predict_quantiles(X_a, quantiles=quantiles)

        yq_ar = np.rollaxis(yq_a, 2)
        y_q = np.stack(
            [self._coreg_data_inverse_transform_y(y) for y in yq_ar],
            axis=2
        )

        #from IPython import embed; embed()
        return y_q

    def _coreg_data_transform_x(self, X):
        """ Augment the input data X with an extra column containing indices
            for each output dim.

            For a model with N dependent variables:

                X_a = [[X], [0],
                       [X], [1],
                       ..., ...,
                       [X], [N]]

                X_a.shape = (X.shape[0] * N, X.shape[1] + 1)
        """
        assert X.shape[1] == self.input_dim
        X_a = np.vstack([np.hstack([X, np.full((X.shape[0], 1), i)]) for i in
                         range(self.output_dim)])
        return X_a

    def _coreg_data_transform_y(self, y):
        """ Reshape y data into a single column. """
        assert y.shape[1] == self.output_dim
        y_a = y.reshape((-1, 1), order='F')
        return y_a

    def _coreg_data_inverse_transform_y(self, y_a):
        """ Reshape y data back into [n x m] where n is number of samples and
            m is number of targets.
        """
        y = y_a.reshape((-1, self.output_dim), order='F')
        return y
