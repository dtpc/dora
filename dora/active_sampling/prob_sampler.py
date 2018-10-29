import numpy as np
from .acquisition_functions import UpperBound
from abc import ABCMeta, abstractmethod


class Sampler(metaclass=ABCMeta):

    def __init__(self, lower, upper):
        assert len(lower) == len(upper)
        self.lower = lower
        self.upper = upper
        self.ndim = len(lower)

    def random_sample(self, n=1):
        """ Return `n` random samples of X, within upper/lower bounds. """
        X = np.random.uniform(self.lower, self.upper, (n, self.ndim))
        return X

    @abstractmethod
    def pick_from(self, X):
        """ Return the index of the feature point `X` to observe next, and the
            expected value of the observation.
        """
        i = None
        y_exp = None
        return i, y_exp

    def pick(self, n_test=500):
        """ Return the feature point to observe next and the expected value
            of the observation.
        """
        Xq = self.random_sample(n_test)
        i, y = self.pick_from(Xq)
        x = Xq[i, :]
        return x, y


class ProbSampler(Sampler):
    """ Sampler that utilises a probabilistic model. Model must have
        functions 'fit()' and 'predict_proba()' (which returns a mean and
        variance).
    """

    def __init__(self, lower, upper, model, acq_fn=UpperBound(), seed=None):
        Sampler.__init__(self, lower, upper)

        self.model = model
        self.acq_fn = acq_fn

        if seed:
            np.random.seed(seed)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.pending_results = {}
        self.virtual_flag =
        self.model.fit(X, y)

    def pick_from(self, xs):
        a, y_exp, _ = self._eval_acq_fn(xs)
        i = np.argmax(a)
        y = y_exp[i, :]
        return i, y

    def update(self, uid, y_obs):
        pass

    def remove(self, uid):
        """ Remove an expected observation from the data set. """
        pass

    def predict(self, xs):
        """ Return the expected value and variance at the points in `xs`. """
        y_exp, y_var = self.model.predict_proba(xs)
        return y_exp, y_var

    def _eval_acq_fn(self, xs, acq_fn=None):
        """ Evaluate the acquisition function at the points `xs`. """
        fn = acq_fn or self.acq_fn
        y_exp, y_var = self.predict(xs)
        a = fn(y_exp, y_var)
        return a, y_exp, y_var

    def eval_acq_fn(self, xs, acq_fn=None):
        """ Evaluate the acquisition function at the points `xs`. """
        a, _, _ = self._eval_acq_fn(xs, acq_fn)
        return a


