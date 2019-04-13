# BSD 3-Clause License
#
# Copyright (c) 2019, Robert A. Milton
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Contains base classes for various models.

Because implementation may involve parallelization, these classes should only contain pre-processing and post-processing.


Contents:
    :AnisotropicStationaryKernel(ABC): Abstract interface class.
    :Model(ABC): Abstract base class.
    :GaussianProcess(Model): Abstract interface class.
    :GaussianBundle(Model): Abstract interface class.
"""

from romcomma.typing_ import PathLike, Optional, NamedTuple, NP, Tuple, List, Type, Union
from pathlib import Path
from romcomma.data import Frame, Fold
from collections import namedtuple
from numpy import zeros, einsum, reshape, eye, pi, log, trace, exp, divide, require, diagflat, full
from numpy.linalg import slogdet
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from romcomma.model import base


# noinspection PyPep8Naming
class AnisotropicExponentialQuadratic(base.Kernel):
    """ AnisotropicExponentialQuadratic Kernel.

    ``Parameters = namedtuple("Parameters", ['anisotropy'])``

    ``parameters.anisotropy`` is the precision (inverse covariance) matrix used to calculate
    ``exponent = transpose(XT) * parameters.anisotropy * XT``.
    """

    MEMORY_LAYOUT = 'F'

    Parameters = namedtuple("Parameters", ['anisotropy'])

    def calculate(self):
        if self._parameters.anisotropy.shape[0] == 1:  # parameters.anisotropy.shape == (1,L,1,L)
            metric = einsum("nom, ikjl, nom -> nokl", self._DX, self._parameters.anisotropy, self._DX,
                            dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        elif self._parameters.anisotropy.shape[2] == 1:  # parameters.anisotropy.shape == (M,L,1,L)
            metric = einsum("nom, mkil, nom -> nokl", self._DX, self._parameters.anisotropy, self._DX,
                            dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        else:  # parameters.anisotropy.shape == (M,L,M,L)
            metric = einsum("noi, ikml, nom -> nokl", self._DX, self._parameters.anisotropy, self._DX,
                            dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        self._tensor4[:] = exp(divide(metric, -2))

    @property
    def parameters(self) -> Parameters:
        """ ``parameters.anisotropy.shape = (L, (1 or M), L, (1 or M))``.
        Factored by ``L``, ``parameters.anisotropy`` is the upper triangular cholesky decomposition of sigma_x. """
        return self._parameters

    @parameters.setter
    def parameters(self, value: Parameters):
        """ ``parameters.anisotropy.shape = (L, (1 or M), L, (1 or M))``.
        Factored by ``L``, ``parameters.anisotropy`` is the upper triangular cholesky decomposition of sigma_x. """
        assert value.anisotropy.shape[0] == value.anisotropy.shape[2] == self._L
        self._parameters = value
        self.calculate()

    # noinspection PyUnusedLocal
    def __init__(self, X0: NP.Matrix, X1: NP.Matrix, L: int):
        """ Construct an AnisotropicExponentialQuadratic(Kernel).

        Args:
            X0: An ``N0xM`` Design (feature) matrix.
            X1: An ``N1xM`` Design (feature) matrix.
            L: The output (label) dimensionality. The _kernel must ultimately produce a tensor4 of shape N0xN1xLxL
        """
        self._X0 = X0
        self._X1 = X1
        self._L = L
        self._N0, self._M = self._X0.shape
        self._N1 = self._X1.shape[0]
        self._DX = (reshape(self._X0, (self._N0, 1, self._M), order=self.MEMORY_LAYOUT) -
                    reshape(self._X1, (1, self._N1, self._M), order=self.MEMORY_LAYOUT))
        self._tensor4 = zeros(self.tensor_shape, dtype=float, order=self.MEMORY_LAYOUT)
        self._parameters = None


# noinspection PyPep8Naming
class GaussianProcess(base.GaussianProcess):
    """ Pure abstract interface to a Gaussian Process. This assumes a minimal

    ``Parameters = namedtuple("Parameters", ['anisotropy', 'f', 'e'])``

    which must be overridden if derived classes need to incorporate extra parameters.
    """

    MEMORY_LAYOUT = 'F'

    Parameters = namedtuple("Parameters", ['f', 'e'])

    @property
    def f_derivative(self) -> NP.Matrix:
        result = zeros((self._L, self._L), dtype=float, order=self.MEMORY_LAYOUT)
        kernel_matrix = self.kernel.tensor4.reshape((self._N, self._N, self.kernel.L, self.kernel.L), dtype=float, order=self.MEMORY_LAYOUT)
        if self.kernel.L == 1:
            for l in range(self._L):
                result[l, l] = einsum("ij, ji -> ",
                                      self._derivative_factor[l * self._N:(l + 1) * self._N, l * self._N:(l + 1) * self._N],
                                      kernel_matrix, dtype=float, order=self.MEMORY_LAYOUT, optimize=True) / 2
                for k in range(l + 1, self._L):
                    result[l, k] = einsum("ij, ji -> ",
                                          self._derivative_factor[k * self._N:(k + 1) * self._N, l * self._N:(l + 1) * self._N],
                                          kernel_matrix, dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        else:
            for l in range(self._L):
                result[l, l] = einsum("ij, ji -> ",
                                      self._derivative_factor[l * self._N:(l + 1) * self._N, l * self._N:(l + 1) * self._N],
                                      kernel_matrix[l * self._N:(l + 1) * self._N, l * self._N:(l + 1) * self._N],
                                      dtype=float, order=self.MEMORY_LAYOUT, optimize=True) / 2
                for k in range(l + 1, self._L):
                    result[l, k] = einsum("ij, ji -> ",
                                          self._derivative_factor[k * self._N:(k + 1) * self._N, l * self._N:(l + 1) * self._N],
                                          kernel_matrix[l * self._N:(l + 1) * self._N, k * self._N:(k + 1) * self._N],
                                          dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        return result

    @property
    def e_derivative(self) -> NP.Matrix:
        result = zeros((self._L, self._L), dtype=float, order=self.MEMORY_LAYOUT)
        for l in range(self._L):
            result[l, l] = trace(self._derivative_factor[l * self._N:(l + 1) * self._N, l * self._N:(l + 1) * self._N]) / 2
            for k in range(l + 1, self._L):
                result[l, k] = trace(self._derivative_factor[k * self._N:(k + 1) * self._N, l * self._N:(l + 1) * self._N])
        return result

    def predict(self, x: NP.Matrix, y_instead_of_f: bool=True) -> Tuple[NP.Vector, NP.Matrix]:
        """ Predicts the response to input x.

        Args:
            x: An ``NxM`` Design (feature) matrix.
            y_instead_of_f: True to include noise in the result.
        Returns: The distribution of y or f, as a (mean_vector, covariance_matrix) tuple.
        """
        n = x.shape[0]
        variance = self.KernelType(x, x, self.kernel.L)
        covariance = self.KernelType(self.kernel.X0, x, self.kernel.L)
        variance.parameters = self.kernel.parameters
        covariance.parameters = self.kernel.parameters
        if self.kernel.L == 1:
            variance = einsum("kl, noij -> nokl", self._parameters.f, variance.tensor4, dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
            covariance = einsum("kl, noij -> nkol", self._parameters.f, covariance.tensor4, dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        else:
            variance = einsum("kl, nokl -> nokl", self._parameters.f, variance.tensor4, dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
            covariance = einsum("kl, nokl -> nkol", self._parameters.f, covariance.tensor4, dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        if y_instead_of_f:
            for k in range(self._L):
                for l in range(self._L):
                    variance[:, :, k, l] += self._parameters.e[k, l]
        variance = einsum("nokl -> nkol", variance, dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        variance.shape = (n * self._L, n * self._L)
        covariance.shape = (self._N * self._L, n * self._L)
        return (einsum("ij, il -> jl", covariance, self._inv_prior_Y_y, dtype=float, order=self.MEMORY_LAYOUT, optimize=True),
                variance - einsum("jk, ji, il -> kl", covariance, self._prior_Y_inv, covariance,
                                  dtype=float, order=self.MEMORY_LAYOUT, optimize=True),)

    @property
    def log_marginal_likelihood(self) -> float:
        return -(self._N * self._L * log(2 * pi) - slogdet(self._prior_Y)[-1] +
                 einsum("il, ik -> lk", self._Y, self._inv_prior_Y_y, dtype=float, order=self.MEMORY_LAYOUT, optimize=True)[0, 0]) / 2

    @property
    def posterior_Y(self) -> Tuple[NP.Vector, NP.Matrix]:
        return (einsum("ij, il -> jl", self._prior_F, self._inv_prior_Y_y, dtype=float, order=self.MEMORY_LAYOUT, optimize=True),
                self._prior_Y - einsum("jk, ji, il -> kl", self._prior_F, self._prior_Y_inv, self._prior_F,
                                       dtype=float, order=self.MEMORY_LAYOUT, optimize=True),)

    @property
    def posterior_F(self) -> Tuple[NP.Vector, NP.Matrix]:
        return (einsum("ij, il -> jl", self._prior_F, self._inv_prior_Y_y, dtype=float, order=self.MEMORY_LAYOUT, optimize=True),
                self._prior_F - einsum("jk, ji, il -> kl", self._prior_F, self._prior_Y_inv, self._prior_F,
                                       dtype=float, order=self.MEMORY_LAYOUT, optimize=True),)

    def calculate(self):
        self._prior_F.shape = self._prior_Y.shape = self.tensor_shape
        if self.kernel.L == 1:
            self._prior_F[:] = einsum("kl, noij -> nkol", self._parameters.f, self.kernel.tensor4,
                                      dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        else:
            self._prior_F[:] = einsum("kl, nokl -> nkol", self._parameters.f, self.kernel.tensor4,
                                      dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        self._prior_Y[:] = einsum("no, kl -> nkol", self._eye_minor, self._parameters.e,
                                  dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        self._prior_Y[:] += self._prior_F
        self._prior_F.shape = self._prior_Y.shape = self.matrix_shape
        (prior_Y_cho, lower) = cho_factor(self._prior_Y, lower=False, overwrite_a=False, check_finite=False)
        self._prior_Y_inv[:] = cho_solve((prior_Y_cho, lower), self._eye, overwrite_b=False, check_finite=False)
        self._inv_prior_Y_y[:] = einsum("ij, il -> jl", self._prior_Y_inv, self._Y,
                                        dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
        self._derivative_factor[:] = einsum("il, jl -> ij", self._inv_prior_Y_y, self._inv_prior_Y_y,
                                            dtype=float, order=self.MEMORY_LAYOUT, optimize=True) + self._prior_Y_inv

    @property
    def kernel_parameters(self) -> base.Kernel.Parameters:
        return self.kernel.parameters

    @kernel_parameters.setter
    def kernel_parameters(self, value: base.Kernel.Parameters):
        """ Triggers update of Kernel, NOT GP."""
        self.kernel.parameters = value

    # noinspection PyUnusedLocal
    def __init__(self, X: NP.Matrix, Y: NP.Matrix, kernel_per_f: bool, kernel: Union[Type[base.Kernel], base.Kernel],
                 kernel_parameters: base.Kernel.Parameters = None):
        """ Construct a GaussianProcess.

        Args:
            X: An ``NxM`` Design (feature) matrix.
            Y: An ``NxL`` Response (label) matrix.
            kernel_per_f: If True, the _kernel depends on output dimension (l) as well as input dimension (m).
            kernel: The Kernel Type, derived from Kernel, or an existing _kernel of this type.
            kernel_parameters: The parameters with which to initialize the _kernel (optional)
        """
        self._N, self._L = Y.shape
        if isinstance(kernel, base.Kernel):
            self.kernel = kernel
        else:
            self.kernel = kernel(X, X, self._L) if kernel_per_f else kernel(X, X, 1)
        if kernel_parameters is not None:
            self.kernel.parameters = kernel_parameters
        self._Y = Y.reshape(self.vector_shape, order=self.MEMORY_LAYOUT)
        self._prior_F = zeros(self.tensor_shape, dtype=float, order=self.MEMORY_LAYOUT)
        self._prior_Y = zeros(self.tensor_shape, dtype=float, order=self.MEMORY_LAYOUT)
        self._prior_Y_inv = zeros(self.matrix_shape, dtype=float, order=self.MEMORY_LAYOUT)
        self._inv_prior_Y_y = zeros(self.vector_shape, dtype=float, order=self.MEMORY_LAYOUT)
        self._derivative_factor = zeros(self.matrix_shape, dtype=float, order=self.MEMORY_LAYOUT)
        self._eye = eye(*self.matrix_shape, dtype=float, order=self.MEMORY_LAYOUT)
        self._eye_minor = eye(self._N, dtype=float, order=self.MEMORY_LAYOUT)
        self._parameters = None


# noinspection PyPep8Naming
class GaussianBundle(base.GaussianBundle):
    """ Interface to a Gaussian Process."""

    MEMORY_LAYOUT = 'F'

    # noinspection PyUnusedLocal
    def optimize(self, **kwargs):
        pass

    def predict(self, X: NP.Matrix) -> List[Tuple[NP.Matrix, NP.Matrix]]:
        pass

    # noinspection PyUnresolvedReferences
    def _validate_parameters(self):
        """ Specific validation. """
        super()._validate_parameters()

    def _full(self, matrix: NP.Matrix) -> NP.Matrix:
        if matrix.shape == (self._L, self._L):
            return matrix
        elif matrix.shape == (self._L, 1):
            return require(diagflat(matrix), dtype=float, requirements=self.MEMORY_LAYOUT)
        else:
            return require(diagflat(full((self._L, 1), matrix[0, 0], dtype=float, order=self.MEMORY_LAYOUT)),
                           dtype=float, requirements=self.MEMORY_LAYOUT)

    def __init__(self, fold: Fold, name: str, parameters: Optional[base.GaussianBundle.Parameters] = None, overwrite: bool = False):
        """ Constructor.

        Args:
            fold: The location of the Model.GaussianProcess. Must be a fold.
            name: The name of this Model.GaussianProcess
            parameters: The model parameters. If ``None`` these are read from ``fold/name``. Otherwise these are written to ``fold/name``.
                Each parameter is a covariance matrix, provided as a square Matrix, or a Vector if diagonal.
            overwrite: If True, any existing directory named ``fold/name`` is deleted. Otherwise no existing files are overwritten.
        """
        super().__init__(fold, name, parameters, overwrite)
        anisotropy = require(self._parameters.anisotropy.df.values, dtype=float, requirements=[self.MEMORY_LAYOUT])
        anisotropy.shape = self._anisotropy_shape
        self._f = require(self._parameters.f.df.values, dtype=float, requirements=[self.MEMORY_LAYOUT])
        self._e = require(self._parameters.e.df.values, dtype=float, requirements=[self.MEMORY_LAYOUT])
        X = require(self._fold.X.values, dtype=float, requirements=[self.MEMORY_LAYOUT])
        Y = require(self._fold.Y.values, dtype=float, requirements=[self.MEMORY_LAYOUT])
        self._independent_gps = (max(self._f.shape, self._e.shape) != (self._L, self._L))
        if self._independent_gps:
            if self._kernel_per_f:
                self._gp = [GaussianProcess(X, Y, True, self._KernelType, self._KernelType.Parameters(anisotropy=anisotropy[:, [l], :, [1]]))
                            for l in range(self._L)]
            else:
                shared_kernel = self._KernelType(X, X, 1)
                shared_kernel.parameters = self._KernelType.Parameters(anisotropy=anisotropy)
                self._gp = [GaussianProcess(X, Y, False, shared_kernel) for l in range(self._L)]
            for l in range(self._L):
                self._gp[l].parameters = GaussianProcess.Parameters(f=self._f[[min(l, self._f.shape[0])], [1]],
                                                                    e=self._e[[min(l, self._e.shape[0])], [1]])
        else:
            self._gp = [GaussianProcess(X, Y, self._kernel_per_f, self._KernelType, self._KernelType.Parameters(anisotropy=anisotropy))]
            self._gp[0].parameters = GaussianProcess.Parameters(f=self._full(self._f), e=self._full(self._e))
