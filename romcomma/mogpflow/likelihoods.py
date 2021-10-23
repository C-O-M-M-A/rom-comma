#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2021 Robert A. Milton. All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
# 
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Contains extensions to gpflow.likelihoods."""

from __future__ import annotations

from typing import Tuple
from romcomma.mogpflow.base import Covariance

from gpflow.config import default_float
from gpflow.likelihoods import QuadratureLikelihood
from gpflow.logdensities import multivariate_normal
import tensorflow as tf


class MOGaussian(QuadratureLikelihood):
    """ A non-diagonal, multivariate likelihood, extending gpflow. The code is the multivariate version of gf.likelihoods.Gaussian.

    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-3 is therefore imposed on the
    likelihood Covariance.cholesky_diagonal elements by default.
    """

    def __init__(self, variance, **kwargs):
        """ Constructor, which passes the Cholesky decomposition of the variance matrix.

        Args:
            variance: The covariance matrix of the likelihood, expressed in tensorflow or numpy. Is checked for symmetry and positive definiteness.
            **kwargs: Keyword arguments forwarded to :class:`Likelihood`.
        """
        self._covariance = Covariance(variance)
        super().__init__(*(self._covariance.shape))

    @property
    def covariance(self):
        """ A mogpflow.base.Covariance, which is the covariance matrix of the likelihood distribution."""
        return self._covariance

    def N(self, data) -> int:
        """ The number of datapoints in data, assuming the last 2 dimensions have been concatenated to LN. """
        return int(data.shape[-1] / self.latent_dim)

    def split_axis_shape(self, data) -> Tuple[int, int]:
        """ Split the final data axis length LN into the pair (L,N). """
        return self.latent_dim, self.N(data)

    def add_variance(self, Fvar) -> tf.Tensor:
        assert tf.rank(Fvar) == 2, f'mogpflow.Likelihood only accepts Fvar of rank 2 at present, provided Fvar of rank {tf.rank(Fvar)}.'
        return tf.reshape(self._covariance.variance + tf.reshape(Fvar, self.split_axis_shape(Fvar) * 2), Fvar.shape)

    def _log_prob(self, F, Y):
        return tf.reduce_sum(multivariate_normal(tf.reshape(Y, self.split_axis_shape(Y)),
                                                 tf.reshape(F, self.split_axis_shape(F)),
                                                 self._covariance.cholesky))

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F):
        return tf.reshape(tf.broadcast_to(self._covariance.variance, self.split_axis_shape(F) * 2), F.shape * 2)

    def _predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), self.add_variance(Fvar)

    def _predict_log_density(self, Fmu, Fvar, Y):
        return tf.reduce_sum(multivariate_normal(Y, Fmu, tf.linalg.cholesky(self.add_variance(Fvar))))

    def _variational_expectations(self, Fmu, Fvar, Y):
        tr = tf.linalg.cholesky_solve(tf.linalg.cholesky(self._conditional_variance(Fmu)), Fvar)
        return self._log_prob(Fmu, Y) - 0.5 * tf.linalg.trace(tr)
