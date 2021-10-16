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

""" Contains extensions to gpflow.kernels."""

from __future__ import annotations

from romcomma.mogpflow.base import Covariance
from abc import abstractmethod
from gpflow.config import default_float
from gpflow.kernels import Kernel, AnisotropicStationary
from gpflow import Parameter, set_trainable
from gpflow.utilities import positive
from gpflow.models.util import data_input_to_tensor
import tensorflow as tf
import numpy as np


class Stationary(AnisotropicStationary, Kernel):
    """
    Base class for stationary kernels, i.e. kernels that only
    depend on

        d = x - x'

    Derived classes should implement K_d(self, d): Returns the kernel evaluated
    on d, which is the pairwise difference matrix, scaled by the lengthscale
    parameter ℓ (i.e. [(X - X2ᵀ) / ℓ]). The last axis corresponds to the
    input dimension.
    """

    @property
    def covariance(self):
        """ The covariance matrix as a mogpflow.base.Covariance object."""
        return self._covariance

    @property
    def lengthscales(self):
        """ The kernel lengthscales as an (L,M) matrix."""
        return tf.reshape(self._lengthscales, (self.L, self._M))

    @property
    def is_lengthscales_trainable(self):
        """ Boolean value indicating whether the kernel lengthscales are trainable."""
        return self._lengthscales.trainable

    @is_lengthscales_trainable.setter
    def is_lengthscales_trainable(self, value: bool):
        set_trainable(self._lengthscales, value)

    def K_diag(self, X):
        """ The kernel diagonal.

        Args:
            X: An (N,M) Tensor.
        Returns: An (L, L, N) Tensor.
        """
        assert len(tf.shape(X)) == 2, f'mogpflow.kernels.Stationary currently only accepts inputs X of rank 2, which X.shape={tf.shape(X)} does not obey.'
        return tf.broadcast_to(tf.reshape(self._covariance.value, self._covariance.shape + (1,)), self._covariance.shape + (tf.shape(X)[-2].numpy(),))

    def K_unit_variance(self, X, X2=None):
        return self.K_d_unit_variance(self.scaled_difference_matrix(X, X2))

    @abstractmethod
    def K_d_unit_variance(self, d):
        """ The kernel with variance=ones(). This can be cached during optimisations where only the variance is trainable.

        Args:
            d: An (L,N,L,N,M) Tensor.
        Returns: An (L,N,L,N) Tensor.
        """
        raise NotImplementedError(f'You must implement K_d_unit_variance(self, d) in {type(self)}.')

    def K_d_apply_variance(self, K_d_unit_variance):
        """ Multiply the unit variance kernel by the kernel variance.

        Args:
            K_d_unit_variance: An (L,N,L,N) Tensor.
        Returns: An (L,N,L,N) Tensor
        """
        assert len(tf.shape(K_d_unit_variance)) == 4, f'mogpflow.kernels.Stationary currently only accepts inputs K_d_unit_variance of rank 4, ' \
                                                         f'which K_d_unit_variance.shape={tf.shape(K_d_unit_variance)} does not obey.'
        return self._covariance.variance * K_d_unit_variance

    def K_d(self, d):
        """ The kernel.

        Args:
            d: An (L,N,L,N,M) Tensor.
        Returns: An (L,N,L,N) Tensor.
        """
        return self.K_d_apply_variance(self.K_d_unit_variance(d))

    def __init__(self, variance, lengthscales, name='kernel', active_dims=None):
        """ Kernel Constructor.

        Args:
            variance: An (L,L) symmetric, positive definite matrix for the signal variance.
            lengthscales: An (L,M) matrix of positive lengthscales.
            name: The name of this kernel.
            active_dims: Which of the input dimensions are used. The default None means all of them.
        """
        super(Kernel).__init__(active_dims=active_dims, name=name)  # Do not call gf.kernels.Stationary.__init__()!
        self._covariance = Covariance(value=np.atleast_2d(variance), name=name + '.covariance')
        self._L = self._covariance.shape[0]
        lengthscales_shape = tf.shape(data_input_to_tensor(lengthscales))
        self._M = 1 if lengthscales_shape == () or lengthscales_shape == (self._L,) else lengthscales_shape[-1]
        lengthscales = tf.broadcast_to(lengthscales, (self._L, 1, self._M))
        self._lengthscales = Parameter(lengthscales, transform=positive(), trainable=False, name=name + '.lengthscales')
        self.is_lengthscales_trainable = False
        self._validate_ard_active_dims(self._lengthscales[0, 0])


class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is

        k(d) = σ² exp{-½ r²}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter

    Functions drawn from a GP with this kernel are infinitely differentiable!
    """

    def K_d_unit_variance(self, d):
        return tf.math.exp(-0.5 * tf.einsum('...M, ...M -> ...', d, d))
