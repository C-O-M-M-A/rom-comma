# BSD 3-Clause License
#
# Copyright (c) 2019-2022, Robert A. Milton
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" gpflow implementation of model.base."""

from __future__ import annotations

from romcomma.typing_ import *
from numpy import atleast_2d, zeros, sqrt, array, transpose
import gpflow as gf
from gpflow.base import Parameter
import tensorflow as tf

class Likelihood(gf.likelihoods.QuadratureLikelihood):
    """ A non-diagonal, multivariate likelihood, extending gpflow. The code is the multivariate version of gf.likelihoods.Gaussian.

    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(self, variance_cho_diag: Sequence[float], variance_cho_off_diag: Sequence[float],
                 variance_cho_diag_lower_bound=DEFAULT_VARIANCE_LOWER_BOUND, **kwargs):
        """ Constructor, which passes the Cholesky decomposition of the variance matrix.

        Args:
            variance_cho_diag: The diagonal elements of the noise_variance Cholesky matrix, a sequence of length L.
            variance_cho_off_diag: The off-diagonal elements of the noise_variance Cholesky matrix, a sequence of length L(L-1)/2.
            variance_cho_diag_lower_bound: The lower bound for the variance diagonal.
            **kwargs: Keyword arguments forwarded to :class:`Likelihood`.
        """
        self.__L = len(variance_cho_diag)
        if len(variance_cho_off_diag) != (expected := self.__L * (self.__L - 1) / 2):
            raise IndexError(f'len(noise_variance_cho_off_diag) = {len(variance_cho_off_diag)} instead of {expected} for L={self.__L}.')
        super().__init__(latent_dim=self.__L, observation_dim=self.__L, **kwargs)
        if min(variance_cho_diag) <= variance_cho_diag_lower_bound:
            raise ValueError(f'The variance_cho_diag of the Gaussian Likelihood must be strictly greater than {variance_cho_diag_lower_bound}')

        self.__variance_cho_diag = Parameter(variance_cho_diag, transform=gf.utilities.positive(lower=variance_cho_diag_lower_bound))
        self.__variance_cho_off_diag = Parameter(variance_cho_off_diag)
        self.__variance_cho_diag = self.__variance_cho_diag
        self.__variance_cho_off_diag = self.__variance_cho_off_diag
        self.__splits = [0]
        for i in range(1, self.__L):
            self.__splits.append(self.__splits[-1] + i)

    @property
    def variance_cho(self):
        result = tf.RaggedTensor.from_row_splits(self.__variance_cho_off_diag.numpy(), self.__splits)
        result = result.to_tensor(default_value=0, shape=(self.__L - 1, self.__L))
        result = tf.concat([tf.zeros(0, self.__L), result])
        return tf.linalg.set_diag(result, self.__variance_cho_diag)

    def _log_prob(self, F, Y):
        return logdensities.gaussian(Y, F, self.variance)

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def _predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def _predict_log_density(self, Fmu, Fvar, Y):
        return tf.reduce_sum(logdensities.gaussian(Y, Fmu, Fvar + self.variance), axis=-1)

    def _variational_expectations(self, Fmu, Fvar, Y):
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance,
            axis=-1,
        )

