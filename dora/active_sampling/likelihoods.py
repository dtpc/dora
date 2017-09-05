
from gpflow.likelihoods import Likelihood
from gpflow.likelihoods import probit
from gpflow.param import Param
from gpflow.param import AutoFlow
from gpflow import transforms
import tensorflow as tf
import numpy as np
from inspect import isclass
from types import MethodType


def likelihood_compute(lh):
    """ Wrapper for GPflow `Likelihood` subclasses which adds compute_* methods
        which run tensor flow when called and return results.

        e.g. Create a Beta likelihood with additional compute_* methods and
             compute the log density on actual values.

            beta = likelihood_compute(Beta)(scale=0.5)
            beta.compute_logp(1, 2)
            >> array([ 10.05222934])
    """

    methods = {
        'logp': [(np.float64,), (np.float64,)],
        'conditional_mean': [(np.float64,)],
        'conditional_variance': [(np.float64,)],
        'predict_mean_and_var': [(np.float64,), (np.float64,)],
        'predict_density': [(np.float64,), (np.float64,), (np.float64,)],
        'variational_expectations': [(np.float64,), (np.float64,), (np.float64,)]
    }

    lh_class = lh if isclass(lh) else lh.__class__

    for name, tf_args in methods.items():
        m = getattr(lh_class, name)
        new_m = AutoFlow(*tf_args)(m)

        # bind method to class if `lh` is an instance
        if not isclass(lh):
            new_m = MethodType(new_m, lh)

        setattr(lh, 'compute_' + name, new_m)

    return lh


def dirichlet(alpha, y):
    lnB = tf.reduce_sum(tf.lgamma(alpha)) - tf.lgamma(tf.reduce_sum(alpha))
    return tf.reduce_sum((alpha - 1) * tf.log(y)) - lnB


class Dirichlet(Likelihood):
    """ Reparameterised Dirichlet distribution with mean given by:

            m = sigma(f)

        and a scale parameter determining alpha as follows:

            m = alpha/sum(alpha)
            alpha = scale * m
    """

    def __init__(self, invlink=probit, scale=1.0):
        Likelihood.__init__(self)
        self.scale = Param(scale, transforms.positive)
        self.invlink = invlink

    def logp(self, F, Y):
        mean = self.invlink(F)
        alpha = mean * self.scale
        p = dirichlet(alpha, Y)
        return p

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        mean = self.invlink(F)
        return (mean - tf.square(mean)) / (self.scale * tf.reduce_sum(mean) + 1.)