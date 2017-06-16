import GPflow as gp
from .gpr_cached import GPRCached
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator
from skleanr.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class GPRModel(BaseEstimator):
    """ sklearn-like wrapper around GPRCached, with scaling of X. """

    def __init__(self, kern, mean_fn=None, name='gpr_model'):
        self.kern = kern
        self.mean_fn = mean_fn or gp.mean_functions.Zero()
        self.name = name
        self.scaler = RobustScaler()

    def params(self):
        """ Return fitted models hyper params as dict. """
        check_is_fitted(self, 'gpr')
        params = self.gpr.get_parameter_dict()
        return params

    def fit(self, x, y, params=None):
        """ Fit the feature scaler and GP model. Optimise the model if params
            is not provided. """
        if x.shape[0] != self.kern.input_dim:
            msg = 'Dimensions of `x` ({}) do not match kernel dimensions ({})'
            raise ValueError(msg.format(x.shape[0], self.kern.input_dim))

        x_scaled = self.scaler.fit_transform(x, y)
        self.gpr = GPRCached(x_scaled, y, self.kern, self.mean_fn, self.name)
        if params is None:
            self.gpr.optimize()
        else:
            self.gpr.set_parameter_dict(params)

    def predict_proba(self, x):
        """ Return the GPs expected value and variance at the points in `x`. """
        check_is_fitted(self, 'gpr')

        x_scaled = self.scaler.transform(x)
        y_exp, y_var = self.gpr.predict_y(x_scaled)
        return y_exp, y_var

    def is_fitted(self):
        fitted = True
        try:
            check_is_fitted(self, 'gpr')
        except NotFittedError:
            fitted = False
        return fitted
