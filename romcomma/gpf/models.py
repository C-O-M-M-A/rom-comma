#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2023 Robert A. Milton. All rights reserved.
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

""" Contains extensions to gpflow.models."""

import tensorflow as tf
from typing import Optional
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.logdensities import multivariate_normal
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.util import data_input_to_tensor
from gpflow.conditionals import base_conditional
import romcomma.gpf as mf

class MOGPR(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is given by

    .. math::
       \log p(Y \,|\, \mathbf f) =
            \mathcal N(Y \,|\, 0, \sigma_n^2 \mathbf{I})

    To train the model, we maximise the log _marginal_ likelihood
    w.r.t. the likelihood variance and kernel hyperparameters theta.
    The marginal likelihood is found by integrating the likelihood
    over the prior, and has the form

    .. math::
       \log p(Y \,|\, \sigma_n, \theta) =
            \mathcal N(Y \,|\, 0, \mathbf{KXX} + \sigma_n^2 \mathbf{I})
    """

    @property
    def M(self):
        """ The input dimensionality."""
        return self._M

    @property
    def L(self):
        """ The output dimensionality."""
        return self._L

    @property
    def KXX(self):
        return self.kernel(self._X, self._X) if self.kernel.lengthscales.trainable else self.kernel.K_d_apply_variance(self._K_unit_variance)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        L = tf.linalg.cholesky(self.likelihood.add_to(self.KXX))
        return tf.reduce_sum(multivariate_normal(self._Y, self._mean, L))

    def predict_f(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        Note that full_cov => full_output_cov (regardless of the ordinate given for full_output_cov), to avoid ambiguity.
        """
        full_output_cov = True if full_cov else full_output_cov
        Xnew = tf.reshape(data_input_to_tensor(Xnew), (-1, self._M))
        n = Xnew.shape[0]
        f_mean, f_var = base_conditional(Kmn=self.kernel(self._X, Xnew), Kmm=self.likelihood.add_to(self.KXX), Knn=self.kernel(Xnew, Xnew),
                                         f=self._Y-self._mean, full_cov=True, white=False)
        f_mean += tf.reshape(self.mean_function(Xnew), f_mean.shape)
        f_mean_shape = (self._L, n)
        f_mean = tf.reshape(f_mean, f_mean_shape)
        f_var = tf.reshape(f_var, f_mean_shape * 2)
        if full_output_cov:
            ein = 'LNln -> LlNn'
        else:
            ein = 'LNLn -> LNn'
        f_var = tf.einsum(ein, f_var)
        if not full_cov:
            f_var = tf.einsum('...NN->...N', f_var)
        perm = tuple(reversed(range(tf.rank(f_var))))
        return tf.transpose(f_mean), tf.transpose(f_var, perm)

    def __init__(self, data: RegressionData, kernel: mf.kernels.MOStationary, mean_function: Optional[mf.mean_functions.MOMeanFunction] = None,
                 noise_variance: float = 1.0):
        """

        Args:
            data: Tuple[InputData, OutputData], which determines L, M and N. Both InputData and OutputData must be of rank 2.
            kernel: Must be well-formed, with an (L,L) variance and an (L,M) lengthscales matrix.
            mean_function: Defaults to Zero.
            noise_variance: Broadcast to (diagonal) (L,L) if necessary.
        """
        self._X, self._Y = self.data = data_input_to_tensor(data)
        if (rank := tf.rank(self._X)) != (required_rank := 2):
            raise IndexError(f'X should be of rank {required_rank} instead of {rank}.')
        self._N, self._M = self._X.shape
        self._L = self._Y.shape[-1]
        if (shape := self._Y.shape) != (required_shape := (self._N, self._L)):
            raise IndexError(f'Y.shape should be {required_shape} instead of {shape}.')
        self._Y = tf.reshape(tf.transpose(self._Y), [-1, 1])   # self_Y is now concatenated into an (LN,)-vector
        if tf.shape(noise_variance).numpy != (self._L, self._L):
            noise_variance = tf.broadcast_to(data_input_to_tensor(noise_variance), (self._L, self._L))
            noise_variance = tf.linalg.band_part(noise_variance, 0, 0)
        likelihood = mf.likelihoods.MOGaussian(noise_variance)
        if mean_function is None:
            mean_function = mf.mean_functions.MOMeanFunction(self._L)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=1)
        self._mean = tf.reshape(self.mean_function(self._X), [-1, 1])
        self._K_unit_variance = self.kernel.K_unit_variance(self._X)
