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

""" SciPy implementation of model.base.

Contents:
**Contents**:
    **Kernel.ExponentialQuadratic(base.Kernel)** class of exponential quadratic kernels.

    **_GaussianProcess(base._GaussianProcess)** class of gaussian processes.

    **GaussianBundle(base.GaussianBundle)** class for a collection of GaussianProcesses.

    **Sobol(base.Sobol)** class which calculates Sobol Indices.

    **Rom(base.Rom)** class which optimizes Sobol Indices.
"""

from ROMCOMMA.tst.romcomma.typing_ import Optional, NP, PathLike, NamedTuple, Tuple, Callable, Union, Type, Dict
from ROMCOMMA.tst.romcomma.data import Fold
from ROMCOMMA.tst.romcomma.model import base
from numpy import atleast_2d, einsum, reshape, exp, divide, zeros, trace, log, pi, eye, ravel, concatenate, array, meshgrid, random
from numpy.linalg import slogdet
from scipy.linalg import cho_factor, cho_solve
import shutil
from enum import IntEnum, auto


class Kernel:
    """ This is just a container for Kernel classes. Put all new Kernel classes in here."""

    # noinspection PyPep8Naming,PyPep8Naming
    class ExponentialQuadratic(base.Kernel):
        """ Implements the exponential quadratic kernel for use with romcomma.scipy_."""

        """ Required overrides."""

        MEMORY_LAYOUT = 'F'
        Parameters = NamedTuple("Parameters", [("lengthscale", NP.Matrix)])
        """
            **lengthscale** -- A (1,M) Covector of ARD lengthscales, or a (1,1) RBF lengthscale.
        """
        DEFAULT_PARAMETERS = Parameters(lengthscale=atleast_2d(0.2))

        """ End of required overrides."""

        def calculate(self):
            """ Calculate the (N0,N1) kernel Matrix K(X0,X1), and store it in self._matrix."""
            super().calculate()
            if self._is_rbf:
                self._matrix = self._parameters.lengthscale[0, 0] * self._DX
            else:
                self._matrix = einsum("m, nom -> no", self._parameters.lengthscale[0, :self._M], self._DX, dtype=float,
                                      order=self.MEMORY_LAYOUT, optimize=True)
            self._matrix = exp(divide(self._matrix, -2))

        def __init__(self, X0: Optional[NP.Matrix], X1: Optional[NP.Matrix], dir_: PathLike = "", parameters: Optional[Parameters] = None):
            """ Construct a Kernel.

            Args:
                X0: An N0xM Design (feature) Matrix. Use None if and only if kernel is only for recording parameters.
                X1: An N1xM Design (feature) Matrix. Use None if and only if kernel is only for recording parameters.
                dir_: The kernel file location. If and only if this is empty, kernel.with_frames=False
                parameters: The kernel parameters. If None these are read from dir_.

            Raises:
                AssertionError: If X0 and X1 have differing numbers of columns.
                AssertionError: If self.parameters.lengthscale is incompatible with self.M.
            """
            super().__init__(X0, X1, dir_, parameters)
            self._is_rbf = (self.parameters.lengthscale.shape[1] == 1)
            if not self.with_frames:
                assert (self._is_rbf or self._parameters.lengthscale.shape[1] >= self._M), \
                    "This ARD kernel has {0:d} lengthscale parameters when M={1:d}.".format(self._parameters.lengthscale.shape[1], self._M)
                self._DX = (reshape(self._X0, (self._N0, 1, self._M), order=self.MEMORY_LAYOUT) -
                            reshape(self._X1, (1, self._N1, self._M), order=self.MEMORY_LAYOUT))
                if self._is_rbf:
                    self._DX = einsum("nom, nom -> no", self._DX, self._DX, dtype=float, order=self.MEMORY_LAYOUT, optimize=True)
                else:
                    self._DX = einsum("nom, nom -> nom", self._DX, self._DX, dtype=float, order=self.MEMORY_LAYOUT, optimize=True)


# noinspection PyPep8Naming
class GP(base.GP):
    """ Implementation of a Gaussian Process."""

    """ Required overrides."""

    MEMORY_LAYOUT = 'F'

    Parameters = NamedTuple("Parameters", [('kernel', NP.Matrix), ('e_floor', NP.Matrix), ('f', NP.Matrix), ('e', NP.Matrix),
                                           ('log_likelihood', NP.Matrix)])
    """ 
        **kernel** -- A numpy [[str]] identifying the type of Kernel, as returned by gp.kernel.TypeIdentifier(). This is never set externally.
            The kernel parameter, when provided, must be a Kernel.Parameters NamedTuple (not an NP.Matrix!) storing the desired kernel 
            parameters. The kernel is constructed and its type inferred from these parameters.

        **e_floor** -- A numpy [[float]] flooring the magnitude of the noise covariance.

        **f** -- An (L,L) signal covariance matrix.

        **e** -- An (L,L) noise covariance matrix.

        **log_likelihood** -- A numpy [[float]] used to record the log marginal likelihood. This is an output parameter, not input.
    """
    DEFAULT_PARAMETERS = Parameters(kernel=Kernel.ExponentialQuadratic.DEFAULT_PARAMETERS,
                                    e_floor=atleast_2d(1E-12), f=atleast_2d(0.9), e=atleast_2d(0.1), log_likelihood=atleast_2d(None))

    DEFAULT_OPTIMIZER_OPTIONS = {}

    KERNEL_NAME = "kernel"

    """ End of required overrides."""

    @property
    def log_likelihood(self) -> float:
        """ The log marginal likelihood of the training data given the GP parameters."""

    def calculate(self):
        """ Fit the GP to the training data. """
        self._test = None

    def predict(self, X: NP.Matrix, Y_instead_of_F: bool = True) -> Tuple[NP.Matrix, NP.Matrix, NP.Tensor3]:
        """ Predicts the response to input X.

        Args:
            X: An (N,M) design Matrix of inputs.
            Y_instead_of_F: True to include noise e in the result covariance.
        Returns: The distribution of Y or f, as a triplet (mean (N, L) Matrix, std (N, L) Matrix, covariance (N, L, L) Tensor3).
        """

    def optimize(self, **kwargs):
        """ Optimize the GP hyper-parameters.

        Args:
            options: A Dict of implementation-dependent optimizer options, following the format of GP.DEFAULT_OPTIMIZER_OPTIONS.
        """
        if kwargs is None:
            kwargs = self._read_optimizer_options() if self.optimizer_options_json.exists() else self.DEFAULT_OPTIMIZER_OPTIONS
        # OPTIMIZE!!!!!
        self._write_optimizer_options(kwargs)
        self.write_parameters(parameters=self.DEFAULT_PARAMETERS)   # Remember to write optimization results.
        self._test = None   # Remember to reset any test results.

    @property
    def inv_prior_Y_Y(self) -> NP.Matrix:
        """ The (N,L) Matrix (f K(X,X) + e I)^(-1) Y."""

    @property
    def posterior_Y(self) -> Tuple[NP.Vector, NP.Matrix]:
        """ The posterior distribution of Y as a (mean Vector, covariance Matrix) Tuple."""

    @property
    def posterior_F(self) -> Tuple[NP.Vector, NP.Matrix]:
        """ The posterior distribution of f as a (mean Vector, covariance Matrix) Tuple."""

    @property
    def f_derivative(self) -> NP.Matrix:
        """ The derivative d(log_likelihood)/df as a Matrix of the same shape as parameters.f. """

    @property
    def e_derivative(self) -> NP.Matrix:
        """ The derivative d(log_likelihood)/de as a Matrix of the same shape as parameters.e. """

    def _validate_parameters(self):
        """ Generic and specific validation.

        Raises:
            IndexError: (generic) if parameters.kernel and parameters.e_floor are not shaped (1,1).
            IndexError: (generic) unless parameters.f.shape == parameters.e == (1,1) or (L,L).
            IndexError: (specific) unless parameters.f.shape == parameters.e == (1,1).
        """
        super()._validate_parameters()

    def __init__(self, fold: Fold, name: str, parameters: Optional[base.GP.Parameters] = None):
        """ GP Constructor. Calls Model.__Init__ to setup parameters, then checks dimensions.

        Args:
            fold: The Fold housing this GaussianProcess.
            name: The name of this GaussianProcess.
            parameters: The model parameters. If None these are read from fold/name, otherwise they are written to fold/name.
        """
        super().__init__(fold, name, parameters)


# noinspection PyPep8Naming
class _GaussianProcess:
    """ Pure abstract interface to a Gaussian Process. This assumes a minimal

    ``Parameters = namedtuple("Parameters", ['anisotropy', 'f', 'e'])``

    which must be overridden if derived classes need to incorporate extra parameters.
    """

    MEMORY_LAYOUT = 'F'

    Parameters = NamedTuple("Parameters", [('f', NP.Matrix), ('e', NP.Matrix)])

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

    def predict(self, x: NP.Matrix, Y_instead_of_F: bool=True) -> Tuple[NP.Vector, NP.Matrix]:
        """ Predicts the response to input x.

        Args:
            x: An ``NxM`` Design (feature) Matrix.
            Y_instead_of_F: True to include noise in the result.
        Returns: The distribution of y or f, as a (mean Vector, covariance Matrix) Tuple.
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
        if Y_instead_of_F:
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
    def tensor_shape(self) -> Tuple[int, int, int, int]:
        return self._N, self._L, self._N, self._L,

    @property
    def matrix_shape(self) -> Tuple[int, int]:
        return self._N * self._L, self._N * self._L,

    @property
    def vector_shape(self) -> Tuple[int, int]:
        return self._N * self._L, 1,

    # noinspection PyUnusedLocal
    def __init__(self, X: NP.Matrix, Y: NP.Matrix, kernel_per_f: bool, kernel: Union[Type[base.Kernel], base.Kernel],
                 kernel_parameters: base.Kernel.Parameters = None):
        """ Construct a _GaussianProcess.

        Args:
            X: An ``NxM`` Design (feature) Matrix.
            Y: An ``NxL`` Response (label) Matrix.
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
class Sobol(base.Sobol):
    """ Interface to a Sobol' Index Calculator and Optimizer.

    Internal quantities are called variant if they depend on Theta, invariant otherwise.
    Invariants are calculated in the constructor. Variants are calculated in Theta.setter."""

    """ Required overrides."""

    MEMORY_LAYOUT = 'F'

    Parameters = NamedTuple("Parameters", [('Mu', NP.Matrix), ('Theta', NP.Matrix), ('D', NP.Matrix), ('S1', NP.Matrix), ('S', NP.Matrix)])
    """ 
        **Mu** -- A numpy [[int]] specifying the number of input dimensions in the rotated basis u.

        **Theta** -- The (Mu, M) rotation matrix ``U = X Theta.T`` (in design matrix terms), so u = Theta x (in column vector terms).

        **D** -- An (L L, M) Matrix of cumulative conditional variances D[l,k,m] = S[l,k,m] D[l,k,M].

        **S1** -- An (L L, M) Matrix of Sobol' main indices.

        **S** -- An (L L, M) Matrix of Sobol' cumulative indices.
    """
    DEFAULT_PARAMETERS = Parameters(*(atleast_2d(None),) * 5)

    # noinspection PyPep8Naming
    class SemiNorm:
        """Defines a SemiNorm on (L,L) matrices, for use by Sobol.

        Attributes:
            value: The SemiNorm.value function, which is Callable[[Tensor], ArrayLike] so it is vectorizable.
            derivative: The SemiNorm.derivative function, which is Callable[[Matrix], Matrix] as it is not vectorizable.
        """

        DEFAULT_META = {'classmethod': 'element', 'L': 1, 'kwargs': {'row': 0, 'column': 0}}

        @classmethod
        def from_meta(cls, meta: Union[Dict, 'Sobol.SemiNorm']) -> 'Sobol.SemiNorm':
            """ Create a SemiNorm from meta information. New SemiNorms should be registered in this function.

            Args:
                meta: A Dict containing the meta data for function construction. Use SemiNorm.DEFAULT_META as a template.
                    Otherwise, meta in the form of a SemiNorm is just returned verbatim, anything elses raises a TypeError.
            Returns: The SemiNorm constructed according to meta.

            Raises:
                TypeError: Unless meta must is a Dict or a SemiNorm.
                NotImplementedError: if meta['classmethod'] is not recognized by this function.
            """
            if isinstance(meta, Sobol.SemiNorm):
                return meta
            if not isinstance(meta, dict):
                raise TypeError("SemiNorm metadata must be a Dict or a SemiNorm, not a {0}.".format(type(meta)))
            if meta['classmethod'] == 'element':
                return cls.element(meta['L'], **meta['kwargs'])
            else:
                raise NotImplementedError("Unrecognized meta['classmethod'] = '{0}'. ".format(meta['classmethod']) +
                                          "Please implement the relevant @classmethod in Sobol.SemiNorm " +
                                          "and register it in Sobol.SemiNorm.from_meta().")

        @classmethod
        def element(cls, L: int, row: int, column: int) -> 'Sobol.SemiNorm':
            """ Defines a SemiNorm on (L,L) matrices which is just the (row, column) element.
            Args:
                L:
                row:
                column:
            Returns: A SemiNorm object encapsulating the (row, column) element semi-norm on (L,L) matrices.

            Raises:
                ValueError: If row or column not in range(L).
            """
            if not 0 <= row <= L:
                raise ValueError("row {0:d} is not in range(L={1:d}.".format(row, L))
            if not 0 <= column <= L:
                raise ValueError("column {0:d} is not in range(L={1:d}.".format(column, L))
            meta = {'classmethod': 'element', 'L': L, 'kwargs': {'row': row, 'column': column}}
            _derivative = zeros((L, L), dtype=float)
            _derivative[row, column] = 1.0

            def value(D: NP.Tensor) -> NP.ArrayLike:
                return D[row, column]

            def derivative(D: NP.Matrix) -> NP.Matrix:
                return _derivative

            return Sobol.SemiNorm(value, derivative, meta)

        def __init__(self, value: Callable[[NP.Tensor], NP.ArrayLike], derivative: Callable[[NP.Matrix], NP.Matrix], meta: Dict):
            """ Construct a SemiNorm on (L,L) matrices.

            Args:
                value: A function mapping an (L,L) matrix D to a float SemiNorm.value
                derivative: A function mapping an (L,L) matrix D to an (L,L) matrix SemiNorm.derivative = d SemiNorm.value / (d D).
                meta: A Dict similar to SemiNorm.DEFAULT_META, giving precise information to construct this SemiNorm
            """
            self.value = value
            self.derivative = derivative
            self.meta = meta

    DEFAULT_OPTIMIZER_OPTIONS = {'semi_norm': SemiNorm.DEFAULT_META, 'N_exploit': 0, 'N_explore': 0, 'options': {'gtol': 1.0E-12}}
    """ 
        **semi_norm** -- A Sobol.SemiNorm on (L,L) matrices defining the Sobol' measure to optimize against.

        **N_exploit** -- The number of exploratory xi vectors to exploit (gradient descend) in search of the global optimum.
            If N_exploit < 1, only re-ordering of the input basis is allowed.

        **N_explore** -- The number of random_sgn xi vectors to explore in search of the global optimum. 
            If N_explore <= 1, gradient descent is initialized from Theta = Identity Matrix.

        **options** -- A Dict of options passed directly to the underlying optimizer.
    """

    NAME = "sobol"

    @classmethod
    def from_GP(cls, fold: Fold, source_gp_name: str, destination_gp_name: str, Mu: int = -1, read_parameters: bool = False) -> 'Sobol':
        """ Create a Sobol object from a saved GP directory.

        Args:
            fold: The Fold housing the source and destination GPs.
            source_gp_name: The source GP directory.
            destination_gp_name: The destination GP directory. Must not exist.
            Mu: The dimensionality of the rotated input basis u. If this is not in range(1, fold.M+1), Mu=fold.M is used.
            read_parameters: True to store read the existing parameters and store them in self.parameters_read (for information purposes only).

        Returns: The constructed Sobol object
        """
        dst = fold.dir / destination_gp_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src=fold.dir / source_gp_name, dst=dst)
        return cls(gp=GP(fold=fold, name=destination_gp_name), Mu=Mu, read_parameters=read_parameters)

    """ End of required overrides."""


# noinspection PyPep8Naming
class ROM(base.ROM):
    """ Reduce Order Model (ROM) Calculator and optimizer.
    This class is documented through its public properties."""

    """ Required overrides."""

    class GP_Initializer(IntEnum):
        ORIGINAL = auto()
        ORIGINAL_WITH_CURRENT_KERNEL = auto()
        ORIGINAL_WITH_GUESSED_LENGTHSCALE = auto()
        CURRENT = auto()
        CURRENT_WITH_ORIGINAL_KERNEL = auto()
        CURRENT_WITH_GUESSED_LENGTHSCALE = auto()
        RBF = auto()

    MEMORY_LAYOUT = 'F'

    Parameters = NamedTuple("Parameters", [('Mu', NP.Matrix), ('D', NP.Matrix), ('S1', NP.Matrix), ('S', NP.Matrix),
                                           ('lengthscale', NP.Matrix), ('log_likelihood', NP.Matrix)])
    """ 
        **Mu** -- A numpy [[int]] specifying the number of input dimensions in the rotated basis u.

        **D** -- An (L L, M) Matrix of cumulative conditional variances D[l,k,m] = S[l,k,m] D[l,k,M].

        **S1** -- An (L L, M) Matrix of Sobol' main indices.

        **S** -- An (L L, M) Matrix of Sobol' cumulative indices.

        **lengthscale** -- A (1,M) Covector of ARD lengthscales, or a (1,1) RBF lengthscale.

        **log_likelihood** -- A numpy [[float]] used to record the log marginal likelihood.
    """
    DEFAULT_PARAMETERS = Parameters(*(atleast_2d(None),) * 6)

    DEFAULT_OPTIMIZER_OPTIONS = {'iterations': 1, 'guess_identity_after_iteration': 1, 'sobol_optimizer_options': Sobol.DEFAULT_OPTIMIZER_OPTIONS,
                                 'gp_initializer': GP_Initializer.CURRENT_WITH_GUESSED_LENGTHSCALE,
                                 'gp_optimizer_options': GP.DEFAULT_OPTIMIZER_OPTIONS}
    """ 
        **iterations** -- The number of ROM iterations. Each ROM iteration essentially calls Sobol.optimimize(options['sobol_optimizer_options']) 
            followed by GP.optimize(options['gp_optimizer_options'])).

        **sobol_optimizer_options*** -- A Dict of Sobol optimizer options, similar to (and documented in) Sobol.DEFAULT_OPTIMIZER_OPTIONS.

        **guess_identity_after_iteration** -- After this many ROM iterations, Sobol.optimize does no exploration, 
            just gradient descending from Theta = Identity Matrix.

        **reuse_original_gp** -- True if GP.optimize is initialized each time from the GP originally provided.

        **gp_optimizer_options** -- A Dict of GP optimizer options, similar to (and documented in) GP.DEFAULT_OPTIMIZER_OPTIONS.
    """

    @classmethod
    def from_ROM(cls, fold: Fold, name: str, suffix: str = ".0", Mu: int = -1, rbf_parameters: Optional[GP.Parameters] = None) -> 'ROM':
        """ Create a ROM object from a saved ROM directory.

        Args:
            fold: The Fold housing the ROM to load.
            name: The name of the saved ROM to create from.
            suffix: The suffix to append to the most optimized gp.
            Mu: The dimensionality of the rotated input basis u. If this is not in range(1, fold.M+1), Mu=fold.M is used.

        Returns: The constructed ROM object
        """
        optimization_count = [optimized.name.count(cls.OPTIMIZED_GB_EXT) for optimized in fold.dir.glob("name" + cls.OPTIMIZED_GB_EXT + "*")]
        source_gp_name = name + cls.OPTIMIZED_GB_EXT * max(optimization_count)
        destination_gp_name = source_gp_name + suffix
        return cls(name=name,
                   sobol=Sobol.from_GP(fold, source_gp_name, destination_gp_name, Mu=Mu, read_parameters=True),
                   optimizer_options=None, rbf_parameters=rbf_parameters)


    @classmethod
    def from_GP(cls, fold: Fold, name: str, source_gp_name: str, optimizer_options: Dict, Mu: int = -1,
                rbf_parameters: Optional[GP.Parameters] = None) -> 'ROM':
        """ Create a ROM object from a saved GP directory.

        Args:
            fold: The Fold housing the ROM to load.
            name: The name of the saved ROM to create from.
            source_gp_name: The source GP directory.
            Mu: The dimensionality of the rotated input basis u. If this is not in range(1, fold.M+1), Mu=fold.M is used.
            optimizer_options: A Dict of ROM optimizer options.

        Returns: The constructed ROM object
        """
        return cls(name=name,
                   sobol=Sobol.from_GP(fold=fold, source_gp_name=source_gp_name, destination_gp_name=name + ".0", Mu=Mu),
                   optimizer_options=optimizer_options, rbf_parameters=rbf_parameters)

    OPTIMIZED_GB_EXT = ".optimized"
    REDUCED_FOLD_EXT = ".reduced"

    """ End of required overrides."""

