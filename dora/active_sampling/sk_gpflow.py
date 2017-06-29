import GPflow as gp
from sklearn.base import BaseEstimator, RegressorMixin
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

    def __init__(self, kern, **model_args):
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

        self.optimize_args = {
            'method': 'L-BFGS-B',
            'maxiter': 10000
        }

        self.n_targets = None

    def _create_model(self, X, y):
        raise NotImplementedError()

    def fit(self, X, y, param_dict=None, optimize_args=None):
        """ Fit the GP model. Optimise the model if params is not provided. """
        if X.shape[1] != self.kern.input_dim:
            msg = 'Dimensions of `x` ({}) do not match kernel dimensions ({})'
            raise ValueError(msg.format(X.shape[1], self.kern.input_dim))

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
        """ Return the predicted value y at defined quantiles. """
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

    def get_params(self, deep=False):
        """ Get GP parameters in sklearn form. Keyword arg `deep` is ignored."""
        check_is_fitted(self, 'model')
        p_dict = self.model.get_parameter_dict()

        def sk_param_name(name):
            return name.split('.', 1)[1].replace('.', '__')

        params = {sk_param_name(k): v for k, v in p_dict.items()}
        return params

    def set_params(self, **params):
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

    def __repr__(self):
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

    def __init__(self, kern, **model_args):
        super().__init__(kern, **model_args)
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

    def __init__(self, kern, inducing, random_seed=None, **model_args):
        super().__init__(kern, **model_args)
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

    def __init__(self, kern, **model_args):
        super().__init__(kern, **model_args)
        self.gp_class = gp.vgp.VGP

    def _create_model(self, X, y):
        self.model = self.gp_class(X, y, kern=self.kern, **self.model_args)


class SkSVGP(SkGPflowRegressor):

    def __init__(self, kern, inducing, random_seed=None, **model_args):
        super().__init__(kern, **model_args)
        self.gp_class = gp.svgp.SVGP
        self.inducing = inducing
        self.random_seed = random_seed

    def _create_model(self, X, y):
        km = MiniBatchKMeans(n_clusters=self.inducing,
                             random_state=self.random_seed)
        Z = km.fit(X).cluster_centers_
        self.model = self.gp_class(X, y, kern=self.kern, Z=Z,
                                   minibatch_size=100, **self.model_args)


class PipelineX(Pipeline):

    def __init__(self, steps, y_transform=None):
        super().__init__(steps)
        self.y_transform = y_transform
        self.n_targets = None

    def fit(self, X, y, **fit_model_kwargs):
        """ Fit y_transform and model. """
        if self.y_transform:
            self.y_transform.fit(y)
            y = self.y_transform.transform(y)
            self.n_targets = y.shape[1]
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
                raise NotImplementedError
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
            self.y_transform.fit(y)
            yt = self.y_transform.transform(y)

        s = super().score(X, yt)
        return s
