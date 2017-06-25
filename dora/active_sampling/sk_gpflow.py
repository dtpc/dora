import GPflow as gp
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted



class PrintCallback:
    def __init__(self):
        self.ii = 0

    def __call__(self, k):
        if self.ii % 1000 == 0:
            print("GPFlow iteration {}".format(self.ii))
        self.ii += 1


class GPflowRegressor(BaseEstimator):
    """ sklearn-like wrapper around GPflow regressors. """

    def __init__(self, gp_class, kern, **model_args):
        self.gp_class = gp_class
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
            'maxiter': 10000,
            'callback': PrintCallback()
        }

    def fit(self, x, y, param_dict=None, optimize_args=None):
        """ Fit the GP model. Optimise the model if params is not provided. """
        if x.shape[1] != self.kern.input_dim:
            msg = 'Dimensions of `x` ({}) do not match kernel dimensions ({})'
            raise ValueError(msg.format(x.shape[1], self.kern.input_dim))

        # create the GPflow model
        self.model = self.gp_class(x, y, kern=self.kern, **self.model_args)

        if param_dict is None:
            optimize_args = optimize_args or {}
            self.optimize(**optimize_args)
        else:
            self.model.set_parameter_dict(param_dict)

        return self

    def predict_proba(self, x):
        """ Return the GPs expected value and variance at the points in `x`. """
        check_is_fitted(self, 'model')
        y_exp, y_var = self.model.predict_y(x)
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
