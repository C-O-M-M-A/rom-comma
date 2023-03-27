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

""" Contains extensions to gpflow.kernels."""


from romcomma.gpf.base import Variance
from abc import abstractmethod
from gpflow.config import default_float
from gpflow.kernels import Kernel, AnisotropicStationary
from gpflow import Parameter, set_trainable
from gpflow.utilities import positive
from gpflow.models.util import data_input_to_tensor
from gpflow.config import default_float
import tensorflow as tf
import numpy as np


class MOStationary(AnisotropicStationary, Kernel):
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
    def L(self):
        return self._L

    @property
    def M(self):
        return self._M

    @property
    def lengthscales_neat(self):
        """ The kernel lengthscales as an (L,M) matrix."""
        return tf.reshape(self.lengthscales, (self._L, self._M))

    def K_diag(self, X):
        """ The kernel diagonal.

        Args:
            X: An (N,M) Tensor.
        Returns: An (L, N, L, N) Tensor.
        """
        N = tf.shape(X)[-2]
        tf.assert_equal(tf.rank(X), 2, f'mogpflow.kernels.MOStationary currently only accepts inputs X of rank 2, which X.shape={tf.shape(X)} does not obey.')
        return self.variance.broadcast(N)

    def K_unit_variance(self, X, X2=None):
        """ The kernel with variance=ones(). This can be cached during optimisations where only the variance is trainable.

        Args:
            X: An (n,M) Tensor.
            X2: An (N,M) Tensor.
        Returns: An (L,N,L,N) Tensor.
        """
        return self.K_d_unit_variance(self.scaled_difference_matrix(X, X if X2 is None else X2))

    @abstractmethod
    def K_d_unit_variance(self, d):
        """ The kernel with variance=ones(). This can be cached during optimisations where only the variance is trainable.

        Args:
            d: An (L,N,L,N,M) Tensor.
        Returns: An (L,N,L,N) Tensor.
        """
        raise NotImplementedError(f'You must implement K_d_unit_variance(self, d) in {type(self)}.')

    def K_d_apply_variance(self, K_d_unit_variance):
        """ Multiply the unit variance kernel by the kernel variance, and reshape.

        Args:
            K_d_unit_variance: An (L,N,L,N) Tensor.
        Returns: An (LN,LN) Tensor
        """
        tf.assert_equal(tf.rank(K_d_unit_variance), 4, f'mogpflow.kernels.MOStationary currently only accepts inputs K_d_unit_variance of rank 4, ' +
                        f'which K_d_unit_variance.shape={tf.shape(K_d_unit_variance)} does not obey.')
        shape = K_d_unit_variance.shape
        return tf.reshape(self.variance.value_to_broadcast * K_d_unit_variance, (shape[-4] * shape[-3], shape[-2] * shape[-1]))

    def K_d(self, d):
        """ The kernel.

        Args:
            d: An (L,N,L,N,M) Tensor.
        Returns: An (LN,LN) Tensor.
        """
        return self.K_d_apply_variance(self.K_d_unit_variance(d))

    def __call__(self, X, X2, *, full_cov=True, presliced=False):
        return super().__call__(X, X2, full_cov=True, presliced=False)

    def __init__(self, variance, lengthscales, name='Kernel', active_dims=None):
        """ Kernel Constructor.

        Args:
            variance: An (L,L) symmetric, positive definite matrix for the signal variance.
            lengthscales: An (L,M) matrix of positive definite lengthscales.
            is_lengthscales_trainable: Whether the lengthscales of this kernel are trainable.
            name: The name of this kernel.
            active_dims: Which of the input dimensions are used. The default None means all of them.
        """
        super(AnisotropicStationary, self).__init__(name=name, active_dims=active_dims)  # Do not call gf.kernels.AnisotropicStationary.__init__()!
        self.variance = Variance(value=np.atleast_2d(variance), name=name + 'Variance')
        # set_trainable(self.variance._cholesky_lower_triangle, False)
        self._L = self.variance.shape[0]
        lengthscales = data_input_to_tensor(lengthscales)
        lengthscales_shape = tuple(tf.shape(lengthscales).numpy())
        self._M = 1 if lengthscales_shape in((),(1,), (1, 1), (self._L,)) else lengthscales_shape[-1]
        lengthscales = tf.reshape(tf.broadcast_to(lengthscales, (self._L, self._M)), (self._L, 1, self._M))
        self.lengthscales = Parameter(lengthscales, transform=positive(), trainable=False, name=name + 'Lengthscales')
        self._validate_ard_active_dims(self.lengthscales[0, 0])


class RBF(MOStationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is

        k(d) = σ² exp{-½ r²}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter

    Functions drawn from a MOGP with this kernel are infinitely differentiable!
    """

    def K_d_unit_variance(self, d):
        return tf.math.exp(-0.5 * tf.einsum('...M, ...M -> ...', d, d))
