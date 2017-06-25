
from GPflow.likelihoods import Likelihood
from GPflow.likelihoods import probit
from GPflow.param import Param
from GPflow.param import AutoFlow
from GPflow import transforms
import tensorflow as tf
import numpy as np


def likelihood_compute(lh_class):
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

    for name, tf_args in methods.items():
        m = getattr(lh_class, name)
        new_m = AutoFlow(*tf_args)(m)
        setattr(lh_class, 'compute_' + name, new_m)

    return lh_class


def dirichlet(alpha, y):
    lnB = tf.reduce_sum(tf.lgamma(alpha)) - tf.lgamma(tf.reduce_sum(alpha))
    return tf.reduce_sum((alpha - 1) * tf.log(y)) - lnB


class Dirichlet(Likelihood):
    """ Reparameterised Dirichlet distributions with mean given by:

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