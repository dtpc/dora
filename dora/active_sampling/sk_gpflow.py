import GPflow as gp
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


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



class GPflowRegressor(BaseEstimator):
    """ sklearn-like wrapper around GPflow regressors. """

    def __init__(self, kern, **model_args):
        self.kern = kern

        default_args = {
            'mean_function': gp.mean_functions.Constant()
        }

        for k, v in default_args.items():
            if k not in model_args:
                model_args[k] = v

        self.model_args = model_args
        self.name = 'name'

        self.optimize_args = {
            'method': 'L-BFGS-B',
            'maxiter': 10000
        }

    def _create_model(self, X, y):
        raise NotImplementedError()

    def fit(self, X, y, param_dict=None, optimize_args=None):
        """ Fit the GP model. Optimise the model if params is not provided. """
        if X.shape[1] != self.kern.input_dim:
            msg = 'Dimensions of `x` ({}) do not match kernel dimensions ({})'
            raise ValueError(msg.format(X.shape[1], self.kern.input_dim))

        # create the GPflow model
        self._create_model(X, y)

        if param_dict is None:
            optimize_args = optimize_args or {}
            self.optimize(**optimize_args)
        else:
            self.model.set_parameter_dict(param_dict)

        return self

    def predict_proba(self, X):
        """ Return the GPs expected value and variance at the points in `x`. """
        check_is_fitted(self, 'model')
        y_exp, y_var = self.model.predict_y(X)
        return y_exp, y_var

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
        print(res)

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


class SkGPR(GPflowRegressor):

    def __init__(self, kern, **model_args):
        super().__init__(kern, **model_args)
        self.gp_class = gp.gpr.GPR

    def _create_model(self, X, y):
        self.model = self.gp_class(X, y, kern=self.kern, **self.model_args)


class SkVGP(GPflowRegressor):

    def __init__(self, kern, **model_args):
        super().__init__(kern, **model_args)
        self.gp_class = gp.vgp.VGP

    def _create_model(self, X, y):
        self.model = self.gp_class(X, y, kern=self.kern, **self.model_args)


class SkSVGP(GPflowRegressor):

    def __init__(self, kern, inducing, **model_args):
        super().__init__(kern, **model_args)
        self.gp_class = gp.vgp.VGP
        self.inducing = inducing

    def _create_model(self, X, y):
        km = MiniBatchKMeans(n_clusters=self.inducing,
                             random_state=int(self.random_state + 1),
                             minibatch_size=100)
        Z = km.fit(X).cluster_centers_
        self.model = gp.svgp.SVGP(X, y, kern=self.kern, Z=Z, **self.model_args)