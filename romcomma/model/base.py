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

"""Contains base classes for various models.

Because implementation may involve parallelization, these classes should only contain pre-processing and post-processing.

**Contents**:
    **SemiNorm**: Defines a semi-norm and its derivative for use by Sobol.

    **Model(ABC)**: Abstract base class, which handles parameters generically.

    **Kernel(Model)**: Abstract interface class.

    **GaussianProcess(Model)**: Abstract interface class.

    **GaussianBundle(Model)**: Abstract interface class, for a collection of GaussianProcesses

    **Sobol(Model)**: Abstract interface class, to calculate Sobol Indices

    **Rom(Model)**: Abstract interface class, to optimize Sobol Indices
"""

import shutil
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from warnings import warn
from numpy import atleast_2d, sqrt, einsum, exp, vstack, prod, eye, append, transpose, zeros, \
    array, meshgrid, ravel, diag, reshape, full, ones, arange, sum
from numpy.random import randint
from pandas import DataFrame
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve, qr

from romcomma.data import Frame, Fold
from romcomma.typing_ import PathLike, Optional, NamedTuple, NP, Tuple, Type, List, Callable, TypeVar, Union

EFFECTIVELY_ZERO = 1.0E-64

SemiNorm_ = TypeVar("SemiNorm_", bound="SemiNorm")


# noinspection PyPep8Naming
class SemiNorm:
    """Defines a semi-norm and its derivative for use by Sobol. """

    @property
    def value(self) -> Callable[[NP.Tensor], NP.ArrayLike]:
        return self._value

    @property
    def derivative(self) -> Callable[[NP.Matrix], NP.Matrix]:
        return self._derivative

    def __init__(self, value: Callable[[NP.Tensor], NP.ArrayLike], derivative: Callable[[NP.Matrix], NP.Matrix]):
        self._value = value
        self._derivative = derivative

    @classmethod
    def element(cls, L: int, row: int, column: int) -> SemiNorm_:
        """ Defines a SemiNorm on (L,L) matrices which is just the (row, column) element.
        Args:
            L:
            row:
            column:

        Returns: A SemiNorm object encapsulating the (row, column) element semi-norm on (L,L) matrices.
        """

        _derivative = zeros((L, L), dtype=float)
        _derivative[row, column] = 1.0

        def value(S: NP.Tensor) -> NP.ArrayLike:
            return S[row, column]

        def derivative(S: NP.Tensor) -> NP.Matrix:
            return _derivative

        return SemiNorm(value, derivative)


class Model(ABC):
    """ Abstract base class for any model. This base class implements the generic file storage and parameter handling.
    The latter is dealt with by each subclass overriding the Model.Parameters type with its own NamedTuple defining the parameter set it takes.
    """

    CSV_PARAMETERS = {'header': [0]}

    Parameters = namedtuple("Parameters", ['OVERRIDE_THIS_TYPE'])
    PARAMETER_DEFAULTS = None

    @staticmethod
    def name(stem: str = "", lengthscale_trainer: Optional[int] = None):
        stem = stem + "." if stem else "unnamed."
        if lengthscale_trainer is None:
            return stem
        elif lengthscale_trainer < 0:
            return stem + "all"
        else:
            return stem + "{0:d}".format(lengthscale_trainer)

    @classmethod
    def rmdir(cls, _dir: Union[str, Path], ignore_errors: bool = True):
        # noinspection PyTypeChecker
        shutil.rmtree(_dir, ignore_errors=ignore_errors)

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def with_frames(self) -> bool:
        return len(self._dir.parts) > 0

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    @parameters.setter
    def parameters(self, value: Parameters):
        self._parameters = value

    @property
    def parameters_as_matrices(self) -> Parameters:
        return self.Parameters(*(p.df.values if isinstance(p, Frame) else p.values if isinstance(p, DataFrame) else p for p in self._parameters))

    def csv(self, index: int) -> Path:
        # noinspection PyProtectedMember
        return (self._dir / self.PARAMETER_DEFAULTS._fields[index]).with_suffix(".csv")

    def read_frames(self):
        assert self.with_frames
        self._parameters = self.Parameters(*(Frame(self.csv(i), csv_parameters=self.CSV_PARAMETERS)
                                             for i, p in enumerate(self.PARAMETER_DEFAULTS)))

    def write_frames(self, parameters: NamedTuple) -> Parameters:
        assert self.with_frames
        return self.Parameters(*(Frame(self.csv(i), p.df if isinstance(p, Frame) else p if isinstance(p, DataFrame) else DataFrame(atleast_2d(p)))
                                 for i, p in enumerate(parameters)))

    @abstractmethod
    def __init__(self, dir_: PathLike = "", parameters: Optional[Parameters] = None, overwrite: bool = False):
        """ Model constructor, to be called by all subclasses as a matter of priority.
            In case ``parameters is None`` the Model is read from ``dir_``.
            Otherwise the model is constructed from ``parameters`` and written to ``dir_``.

        Args:
            dir_: The Model file location, defaults to "".
            parameters: The model parameters. If ``dir_`` is not "", these are read from dir_ (if ``parameters`` is None)
                or written to dir_ (if ``parameters is not None).
            overwrite: whether to overwrite any existing ``dir_``
        Raises:
            ValueError: If ``dir_`` is "" and ``parameters`` is None
        """
        self._dir = Path(dir_)
        if self.with_frames:
            if parameters is None:
                self.read_frames()
            else:
                self._dir.mkdir(mode=0o777, parents=True, exist_ok=overwrite)
                self._parameters = self.write_frames(parameters)
        elif parameters is not None:
            self._parameters = self.Parameters(*(atleast_2d(p) for p in parameters))
        else:
            raise ValueError("dir_ and parameters are both empty.")


# noinspection PyPep8Naming
class Kernel(Model):
    """ Abstract interface to a Kernel. Essentially this is the code contract with the GaussianProcess and GaussianBundle interfaces. """

    MEMORY_LAYOUT = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-all-input-is-Fortran-layout)."

    Parameters = namedtuple("Parameters", ['OVERRIDE_THIS'])
    PARAMETER_DEFAULTS = None
    NAME = "kernel"

    @classmethod
    def TypeFromParameters(cls, parameters: NamedTuple) -> Type[Model]:
        """ Recognize the Type of a Kernel frm its Parameters.

        Args:
            parameters: A Kernel.Parameters array to recognize.
        Returns:
            The type of Kernel that ``parameters`` defines.
        """
        for kernel_type in Kernel.__subclasses__():
            if isinstance(parameters, kernel_type.Parameters):
                return kernel_type
        raise TypeError("Kernel Parameters array of unrecognizable type.")

    @classmethod
    def type_identifier(cls) -> str:
        return cls.__module__.split('.')[-1] + "." + cls.__name__

    @classmethod
    def TypeFromIdentifier(cls, _type_identifier: str) -> Type[Model]:
        """ Convert a type_identifier to a Kernel Type.

        Args:
            _type_identifier: A string generated by Kernel.type_identifier().
        Returns:
            The type of Kernel that ``_type_identifier`` specifies.
        """
        for kernel_type in cls.__subclasses__():
            if kernel_type.type_identifier() == _type_identifier:
                return kernel_type
        raise TypeError("Kernel.type_identifier() of unrecognizable type.")

    @property
    def M(self) -> int:
        return self._M

    @property
    def N0(self) -> int:
        return self._N0

    @property
    def N1(self) -> int:
        return self._N1

    @property
    def X0(self) -> int:
        return self._X0

    @property
    def X1(self) -> int:
        return self._X1

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    @parameters.setter
    def parameters(self, value: Parameters):
        self._parameters = value
        self.calculate()

    @property
    def matrix(self) -> NP.Matrix:
        return self._matrix

    @abstractmethod
    def calculate(self):
        pass

    # noinspection PyUnusedLocal
    @abstractmethod
    def __init__(self, X0: Optional[NP.Matrix], X1: Optional[NP.Matrix], dir_: PathLike = "", parameters: Optional[NamedTuple] = None,
                 overwrite: bool = False):
        """ Construct a Kernel.

        Args:
            X0: An ``N0xM`` Design (feature) matrix. Use None if _kernel is only for recording ``parameters``.
            X1: An ``N1xM`` Design (feature) matrix. Use None if _kernel is only for recording ``parameters``.
            dir_: The Model file location, defaults to "".
            parameters: The model parameters. If ``dir_`` is not "", these are read from dir_ (if ``parameters`` is None)
                or written to dir_ (if ``parameters is not None).
            overwrite: whether to overwrite any existing ``dir_``
        """
        super().__init__(dir_, parameters, overwrite)
        self._matrix = None
        if X0 is not None and X1 is not None:
            self._X0 = X0
            self._X1 = X1
            self._N0, self._M = self._X0.shape
            self._N1 = self._X1.shape[0]
            assert self._M == self._X1.shape[1], "X0 and X1 have incompatible shapes."


# noinspection PyPep8Naming
class GaussianProcess(Model):
    """ Pure abstract interface to a Gaussian Process."""

    MEMORY_LAYOUT = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-all-input-is-Fortran-layout)."

    Parameters = namedtuple("Parameters", ['kernel', 'lengthscale_trainer', 'e_floor', 'f', 'e', 'log_likelihood'])
    PARAMETER_DEFAULTS = Parameters(*(zeros((0, 0)),) * 6)

    @property
    def X(self) -> NP.Matrix:
        return self._X

    @property
    def Y(self) -> NP.Matrix:
        return self._Y

    @property
    def L(self) -> int:
        return self._L

    @property
    def N(self) -> int:
        return self._N

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    @parameters.setter
    def parameters(self, value: Parameters):
        self._parameters = value
        self.calculate()

    @property
    def tensor_shape(self) -> Tuple[int, int, int, int]:
        return self._N, self._L, self._N, self._L,

    @property
    def matrix_shape(self) -> Tuple[int, int]:
        return self._N * self._L, self._N * self._L,

    @property
    def vector_shape(self) -> Tuple[int, int]:
        return self._N * self._L, 1,

    @property
    @abstractmethod
    def f_derivative(self) -> NP.Matrix:
        pass

    @property
    @abstractmethod
    def e_derivative(self) -> NP.Matrix:
        pass

    @property
    @abstractmethod
    def inv_prior_Y_y(self) -> NP.Vector:
        """
        Returns: An (LN, 1) Vector.
        """
        pass

    @property
    def Y_tilde(self) -> NP.Matrix:
        """
        Returns: An (L, N) Matrix.
        """
        return einsum('LK, KN -> LN', atleast_2d(self.parameters.f),
                      self.inv_prior_Y_y.T.reshape((self.L, self.N), order=self.MEMORY_LAYOUT),
                      dtype=float, order=self.MEMORY_LAYOUT, optimize=True)

    @abstractmethod
    def predict(self, x: NP.Matrix, y_instead_of_f: bool = True) -> Tuple[NP.Vector, NP.Matrix]:
        """ Predicts the response to input x.
        Args:
            x: An nxM Design (feature) matrix.
            y_instead_of_f: True to include noise in the result.
        Returns: The distribution of y or f, as a (mean_vector, covariance_matrix) tuple.
        """
        pass

    @property
    @abstractmethod
    def log_marginal_likelihood(self) -> float:
        pass

    @property
    @abstractmethod
    def posterior_Y(self) -> Tuple[NP.Vector, NP.Matrix]:
        pass

    @property
    @abstractmethod
    def posterior_F(self) -> Tuple[NP.Vector, NP.Matrix]:
        pass

    @abstractmethod
    def calculate(self):
        pass

    @abstractmethod
    # noinspection PyUnusedLocal
    def __init__(self, Y: NP.Matrix, kernel: Kernel, parameters: Optional[Parameters] = None):
        """ Construct a GaussianProcess.
        Args:
            kernel: A pre-constructed Kernel.
            Y: An ``NxL`` Response (label) matrix.
            parameters: A GaussianProcess.Parameters NamedTuple.
        """
        super().__init__(parameters=parameters)
        self._kernel = kernel
        self._X = self._kernel.X0
        self._Y = Y
        self._N = Y.shape[0]
        self._L = Y.shape[1]


# noinspection PyPep8Naming
class GaussianBundle(Model):
    """ Interface to a Gaussian Process."""

    MEMORY_LAYOUT = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-all-input-is-Fortran-layout)."

    Parameters = namedtuple("Parameters", ['kernel', 'lengthscale_trainer', 'e_floor', 'f', 'e', 'log_likelihood'])
    PARAMETER_DEFAULTS = Parameters(*(DataFrame(),) * 6)

    @staticmethod
    def dir_path(fold: Fold, name: str) -> Path:
        return fold.dir / name

    @property
    def fold(self) -> Fold:
        return self._fold

    @property
    def X(self) -> NP.Matrix:
        return self._X

    @X.setter
    def X(self, value: NP.Matrix):
        self._X = value

    @property
    def Y(self) -> NP.Matrix:
        return self._Y

    @property
    def L(self) -> int:
        return self._L

    @property
    def M(self) -> int:
        return self._M

    @property
    def N(self) -> int:
        return self._N

    # noinspection PyUnusedLocal
    @abstractmethod
    def optimize(self, **kwargs):
        self._test = None

    @abstractmethod
    def predict(self, x: NP.Matrix, y_instead_of_f: bool = True) -> NP.Tensor4:
        pass

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    @property
    def gp(self) -> List[GaussianProcess]:
        return self._gp

    @property
    def test_csv(self) -> Path:
        return self._dir / "__test__.csv"

    def test(self) -> Frame:
        if self._test is None:
            self._test = Frame(self.test_csv, self._fold.test.df)
            Y_heading = self._fold.meta['data']['Y_heading']
            predictive_mean = (self._test.df.loc[:, [Y_heading]].copy().rename(columns={Y_heading: "Predictive Mean"}, level=0))
            predictive_std = (self._test.df.loc[:, [Y_heading]].copy().rename(columns={Y_heading: "Predictive Std"}, level=0))
            result = self.predict(self._fold.test_X.values)
            predictive_mean.iloc[:] = result[0]
            predictive_std.iloc[:] = sqrt(result[1])
            self._test.df = self._test.df.join([predictive_mean, predictive_std])
            self._test.write()
        return self._test

    @property
    def Y_tilde(self) -> NP.Matrix:
        """
        Returns: An (L, N) Matrix.
        """
        return vstack([gp.Y_tilde for gp in self._gp])

    # noinspection PyUnresolvedReferences
    def _validate_parameters(self):
        """ Generic validation.

        Raises:
            IndexError: if the shape of a parameter does not translate into a square covariance matrix
            IndexError: if f, e and _kernel are any shape other than ``1x1``, ``1xL``, or ``LxL``.
            IndexError: if f.shape does not factor by _kernel.shape
        """
        for param in self.parameters._asdict().items():
            if param[1].df.shape not in ((1, 1), (1, self._L), (self._L, self._L)):
                raise IndexError("parameters.{0:s}.df.shape={1}, while self.L={2:d}. f.shape must be (1, 1), (1, L) or (L, L)."
                                 .format(param[0], param[1].df.shape, self._L))
        if not (self.parameters.f.df.shape == self.parameters.e.df.shape):
            raise ValueError("GaussianBundle requires f and e parameters to be the same shape.")
        if self.parameters.kernel.df.shape != (1, 1):
            raise IndexError("GaussianBundle only accepts a single kernel, so parameter.kernel.shape must be (1,1), not {0}"
                             .format(self.parameters.kernel.df.shape))
        if self.parameters.lengthscale_trainer.df.shape != (1, 1):
            raise IndexError("GaussianBundle only accept a single lengthscale_trainer, so parameter.lengthscale_trainer.shape must be (1,1), not {0}"
                             .format(self.parameters.lengthscale_trainer.df.shape))
        elif self.parameters.lengthscale_trainer.df.iloc[0, 0] is None:
            self.parameters.lengthscale_trainer.df.iloc[0, 0] = -self._L
        if self.parameters.e_floor.df.shape != (1, 1):
            raise IndexError("GaussianBundle only accept a single e_floor, so parameter.e_floor.shape must be (1,1), not {0}"
                             .format(self.parameters.e_floor.df.shape))

    @abstractmethod
    def __init__(self, fold: Fold, name: str, parameters: Optional[Parameters] = None, overwrite: bool = False, reset_log_likelihood: bool = True):
        """ Constructor. Calls Model.__Init__ to setup parameters, then checks dimensions.

        Args:
            fold: The location of the Model.GaussianProcess. Must be a fold.
            name: The name of this Model.GaussianProcess
            parameters: The model parameters. If ``None`` these are read from ``fold/name``.
                Otherwise these are written to ``fold/name``.
                Each parameter is a covariance matrix, provided as a square Matrix, or a CoVector if diagonal.
            overwrite: If True, any existing directory named ``fold/name`` is deleted.
                Otherwise no existing files are overwritten.
        """
        self._fold = fold
        dir_ = self.dir_path(fold, name)
        self._test = None
        self._X = self._fold.X.values.copy()
        self._Y = self._fold.Y.values.copy()
        self._N, self._M, self._L = (self._fold.meta['data']['N'], self._fold.M, self._fold.meta['data']['L'])
        if parameters is None:
            super().__init__(dir_, parameters, overwrite)
            KT = Kernel.TypeFromIdentifier(self._parameters.kernel.df.iloc[0, 0])
            self._kernel_template = KT(X0=None, X1=None, dir_=dir_ / Kernel.NAME, parameters=None, overwrite=False)
        else:
            KT = Kernel.TypeFromParameters(parameters.kernel[0][0])
            self._kernel_template = KT(X0=None, X1=None, dir_=dir_ / Kernel.NAME, parameters=parameters.kernel[0][0], overwrite=True)
            if reset_log_likelihood:
                parameters = parameters._replace(kernel=atleast_2d(KT.type_identifier()), log_likelihood=zeros((1, self._L), dtype=float,
                                                                                                               order=self.MEMORY_LAYOUT))
            else:
                parameters = parameters._replace(kernel=atleast_2d(KT.type_identifier()))
            # noinspection PyTypeChecker
            super().__init__(dir_, parameters, overwrite)
        self._validate_parameters()
        self._kernel = None
        self._gp = []


# noinspection PyPep8Naming
class Sobol(Model):
    MEMORY_LAYOUT = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-all-input-is-Fortran-layout)."

    Parameters = namedtuple("Parameters", ['m_max', 'Theta', 'D', 'S', 'S1'])
    PARAMETER_DEFAULTS = Parameters(*(DataFrame(),) * 5)
    NAME = "sobol"

    @property
    def gb(self):
        return self._gb

    @property
    def dir(self) -> Path:
        return self.gb.dir / self.NAME

    @property
    def lengthscale(self) -> NP.Array:
        return self._lengthscale

    @property
    def m(self) -> int:
        """
        Returns: The row that xi updates - must lie within range(M)
        """
        return self._m

    @property
    def D(self) -> NP.Tensor3:
        """
        Returns: An (L, L, M) Tensor3 of conditional variances.
        """
        return self._D

    @property
    def S(self) -> NP.Tensor3:
        """
        Returns: An (L, L, M) Tensor3 of Sobol' indices.
        """
        return self.D / self.D[:, :, -1]

    @property
    def S1(self) -> NP.Tensor3:
        """
        Returns: An (L, L, M) Tensor3 of Sobol main effect indices.
        """
        return self._S1

    @property
    def F(self) -> NP.Matrix:
        """
        Returns: An (L, N) Matrix.
        """
        return self._F

    @property
    def F1F(self) -> NP.Matrix:
        """
        Returns: An (L, L) Matrix of E[f] @ E[f].T.
        """
        return self._F1F

    @property
    def Theta_old(self) -> NP.Matrix:
        """
        Returns: An (M, M) orthogonal matrix.
        """
        return self._Theta_old

    @Theta_old.setter
    def Theta_old(self, value: NP.Matrix):
        assert value.shape == (self.gb.M, self.gb.M)
        self._Theta_old = value
        self.Theta = self.Theta_old[:self.m + 1, :].copy(order=self.MEMORY_LAYOUT)

    @property
    def Theta(self) -> NP.Matrix:
        """
        Returns: An (m+1, M) matrix.
        """
        return self._Theta

    # noinspection PyAttributeOutsideInit
    @Theta.setter
    def Theta(self, value: NP.Matrix):
        assert value.shape == (self.m + 1, self.gb.M)
        self._Theta = value
        self._Sigma_partial = einsum('M, kM -> Mk', self.Sigma_diagonal, self.Theta, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        self._Sigma = einsum('mM, Mk -> mk', self.Theta, self._Sigma_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        self._Sigma_cho = cho_factor(self._Sigma, lower=False, overwrite_a=False, check_finite=False)
        self._Sigma_cho_det = prod(diag(self._Sigma_cho[0]))
        self._2I_minus_Sigma_partial = einsum('M, kM -> Mk', 2 - self.Sigma_diagonal, self.Theta, optimize=True, dtype=float,
                                              order=self.MEMORY_LAYOUT)
        self._2I_minus_Sigma = einsum('mM, Mk -> mk', self.Theta, self._2I_minus_Sigma_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        self._2I_minus_Sigma_cho = cho_factor(self._2I_minus_Sigma, lower=False, overwrite_a=False, check_finite=False)
        self._2I_minus_Sigma_cho_det = prod(diag(self._2I_minus_Sigma_cho[0]))
        self._D_const = (self._2I_minus_Sigma_cho_det * self._Sigma_cho_det) ** (-1)
        self._inv_Sigma_Theta = cho_solve(self._Sigma_cho, self.Theta, overwrite_b=False, check_finite=False)
        self._Phi_partial = einsum('M, kM -> Mk', self.Phi_diagonal, self.Theta, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        self._Phi = einsum('mM, Mk -> mk', self.Theta, self._Phi_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        self._Phi_cho = cho_factor(self._Phi, lower=False, overwrite_a=False, check_finite=False)
        self._inv_Phi_inv_Sigma_Theta = cho_solve(self._Phi_cho, self._inv_Sigma_Theta, overwrite_b=False, check_finite=False)
        T_inv_Sigma_T = atleast_2d(einsum('NM, mM, mK, NK -> N', self._T_pre, self.Theta, self._inv_Sigma_Theta, self._T_pre,
                                          optimize=True, dtype=float, order=self.MEMORY_LAYOUT))

        T_inv_Phi_inv_Sigma_T = einsum('NOM, mM, mK, NOK -> NO', self._T_pre_cross, self.Theta, self._inv_Phi_inv_Sigma_Theta, self._T_pre_cross,
                                       optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        T_inv_Sigma_T = transpose(T_inv_Sigma_T) + T_inv_Sigma_T
        self.W = exp(-0.5 * (T_inv_Sigma_T - T_inv_Phi_inv_Sigma_T))
        self._D_plus_F1F = self._D_const * einsum('LN, NO, KO -> LK', self.F, self.W, self.F, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        self._D[:, :, self.m] = self._D_plus_F1F - self.F1F

    @property
    def xi_len(self) -> int:
        return self.gb.M - self.m - 1

    @property
    def xi(self) -> NP.Array:
        return self._xi

    @xi.setter
    def xi(self, value: NP.Array):
        assert value.shape[0] == self.xi_len
        norm = 1 - einsum('m, m -> ', value, value, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        norm = norm if norm > 0 else EFFECTIVELY_ZERO
        self._xi = append(sqrt(norm), value)
        self.Theta[self.m, :] = einsum('k, kM -> M', self.xi, self.Theta_old[self.m:, :],
                                       optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        self.Theta = self.Theta[:self.m + 1, :]

    def _calculate_T_pre_cross_F_F1F(self) -> Tuple[NP.Matrix, ...]:
        T_pre = einsum('NM, M -> NM', self.gb.X, self.T_diagonal, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        T_pre_cross = T_pre.reshape((1, self.gb.N, self.gb.M))
        T_pre_cross = transpose(T_pre_cross, (1, 0, 2)) + T_pre_cross
        F = einsum('NM, NM -> N', T_pre, self.gb.X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        F = sqrt(prod(self.Sigma_diagonal)) * exp(-0.5 * F)
        F = einsum('LN, N -> LN', self.gb.Y_tilde, F, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        F1F = einsum('LN, KO -> LK', F, F, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        return T_pre, T_pre_cross, F, F1F

    def _calculate_diagonals(self) -> Tuple[NP.Array, ...]:
        """
        Returns: (lengthscale**(2) + I)**(-1), (lengthscale**(-2) + I)**(-1), (lengthscale**(-2) + 2I) (lengthscale**(-2) + I)**(-1)
        """
        T_diagonal = self.lengthscale ** 2
        Phi_diagonal = T_diagonal ** (-1)
        T_diagonal += 1.0
        Phi_diagonal += 1.0
        Sigma_diagonal = Phi_diagonal ** (-1)
        return T_diagonal ** (-1), Sigma_diagonal, (2 * Phi_diagonal - 1.0) * Sigma_diagonal

    @staticmethod
    def _random_sgn(N_samples, array_len) -> NP.Matrix:
        if N_samples < 3 ** array_len:
            return randint(3, size=(N_samples, array_len)) - 1
        else:
            values = array([-1, 0, 1])
            # noinspection PyUnusedLocal
            values = [values.copy() for i in range(array_len)]
            result = meshgrid(*values)
            result = [ravel(arr) for arr in result]
            return transpose(vstack(result))

    def optimization_target(self, semi_norm: SemiNorm) -> Callable[[NP.Array], float]:
        def target(xi: NP.Array) -> float:
            self.xi = xi
            return -1E6 * semi_norm.value(self.D[:, :, self.m])

        return target

    @property
    def V(self) -> NP.Tensor3:
        """
        Returns: Tensor3(N, N, M-m,)
        """
        Sigma_factor_transpose = self._Theta_jac - einsum('kmj, kM -> mMj', self._inv_Sigma_jac_Sigma, self.Theta,
                                                          optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        Theta_inv_Sigma_Theta_jac = einsum('mMj, mK -> MKj', Sigma_factor_transpose, self._inv_Sigma_Theta,
                                           optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        T_inv_Sigma_T_jac = einsum('NM, MKj, NK -> Nj', self._T_pre, Theta_inv_Sigma_Theta_jac, self._T_pre,
                                   optimize=True, dtype=float, order=self.MEMORY_LAYOUT).reshape(1, self.gb.N, self.xi_len + 1)
        Phi_factor_transpose = self._Theta_jac - einsum('kmj, kM -> mMj', self._inv_Phi_jac_Phi, self.Theta,
                                                        optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        Theta_inv_Phi_inv_Sigma_Theta_jac = einsum('mMj, mK -> MKj', Phi_factor_transpose, self._inv_Phi_inv_Sigma_Theta,
                                                   optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        Theta_inv_Phi_inv_Sigma_Theta_jac -= einsum('kM, kmj, mK -> MKj', self.Theta, self._inv_Phi_inv_Sigma_jac_Sigma, self._inv_Sigma_Theta,
                                                    optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        T_inv_Phi_inv_Sigma_T_jac = einsum('NOM, MKj, NOK -> NOj',
                                           self._T_pre_cross, Theta_inv_Phi_inv_Sigma_Theta_jac, self._T_pre_cross,
                                           optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        return T_inv_Phi_inv_Sigma_T_jac - (T_inv_Sigma_T_jac + transpose(T_inv_Sigma_T_jac, (1, 0, 2)))

    # noinspection PyAttributeOutsideInit
    def calculate_various_jac(self):
        self._Theta_jac = zeros((self.m + 1, self.gb.M, self.xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
        self._Theta_jac[self.m, :, :] = transpose(self.Theta_old[self.m:self.gb.M, :])
        self._Sigma_jac = einsum('mMj, Mk -> mkj', self._Theta_jac, self._Sigma_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        self._inv_Sigma_jac_Sigma = zeros((self.m + 1, self.m + 1, self.xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
        self._2I_minus_Sigma_jac = einsum('mMj, Mk -> mkj', self._Theta_jac, self._2I_minus_Sigma_partial, optimize=True,
                                          dtype=float, order=self.MEMORY_LAYOUT)
        self._inv_2I_minus_Sigma_jac_2I_minus_Sigma = zeros((self.m + 1, self.m + 1, self.xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
        self._Phi_jac = einsum('mMj, Mk -> mkj', self._Theta_jac, self._Phi_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        self._inv_Phi_jac_Phi = zeros((self.m + 1, self.m + 1, self.xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
        self._inv_Phi_inv_Sigma_jac_Sigma = zeros((self.m + 1, self.m + 1, self.xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
        self._log_D_const_jac = zeros(self.xi_len + 1, dtype=float, order=self.MEMORY_LAYOUT)
        for j in range(self.xi_len + 1):
            self._inv_Sigma_jac_Sigma[:, :, j] = cho_solve(self._Sigma_cho, self._Sigma_jac[:, :, j], overwrite_b=False, check_finite=False)
            self._inv_2I_minus_Sigma_jac_2I_minus_Sigma[:, :, j] = cho_solve(self._2I_minus_Sigma_cho, self._2I_minus_Sigma_jac[:, :, j],
                                                                             overwrite_b=False, check_finite=False)
            self._inv_Phi_jac_Phi[:, :, j] = cho_solve(self._Phi_cho, self._Phi_jac[:, :, j], overwrite_b=False, check_finite=False)
            self._inv_Phi_inv_Sigma_jac_Sigma[:, :, j] = cho_solve(self._Phi_cho, self._inv_Sigma_jac_Sigma[:, :, j],
                                                                   overwrite_b=False, check_finite=False)
            self._log_D_const_jac[j] = (sum(diag(self._inv_2I_minus_Sigma_jac_2I_minus_Sigma[:, :, j])) +
                                        sum(diag(self._inv_Sigma_jac_Sigma[:, :, j])))

    @property
    def D_jac(self) -> NP.Tensor3:
        """
        Returns: Tensor3(L, L, M-m)
        """
        self.calculate_various_jac()
        D_derivative = self._D_const * einsum('LN, NO, NOj, KO -> LKj', self.F, self.W, self.V, self.F,
                                              optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        D_derivative -= einsum('j, LK -> LKj', self._log_D_const_jac, self._D_plus_F1F, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        result = -self.xi[1:] / self.xi[0]
        result = einsum('LK, j -> LKj', D_derivative[:, :, 0], result, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        return result + D_derivative[:, :, 1:]

    def optimization_jac(self, semi_norm: SemiNorm) -> Callable[[NP.Array], NP.Array]:
        def jac(xi: NP.Array) -> NP.Array:
            """
            Returns: Array(M-m,)
            """
            self.xi = xi
            return -einsum('LK, LKj -> j', semi_norm.derivative(self.D[:, :, self.m]), self.D_jac,
                           optimize=True, dtype=float, order=self.MEMORY_LAYOUT)

        return jac

    def _optimize_one(self, semi_norm: SemiNorm, N_search: int, N_optimize: int, xi: Optional[NP.Array], **kwargs):
        if xi is None:
            search = self._random_sgn(N_search, self.xi_len) * (self.xi_len + 1) ** (-1 / 2)
        else:
            search = atleast_2d(xi)[:, :self.xi_len].copy()
            assert search.shape == (1, self.xi_len)
        # noinspection PyUnusedLocal
        best = [[0, search[0]] for i in range(N_optimize)]
        for xi in search:
            self.xi = xi
            measure = semi_norm.value(self.D[:, :, self.m])
            for i in range(N_optimize):
                if measure > best[i][0]:
                    for j in reversed(range(i + 1, N_optimize)):
                        best[j] = best[j - 1]
                    best[i] = [measure, xi]
                    break
        for record in best:
            result = optimize.minimize(self.optimization_target(semi_norm), record[1], method='BFGS', jac=self.optimization_jac(semi_norm), **kwargs)
            if result.fun < best[0][0]:
                best[0] = [result.fun, result.x]
        self.xi = best[0][1]

    def _optimal_shuffle(self, semi_norm: SemiNorm):
        xi_temp = zeros(self.xi_len, dtype=float, order=self.MEMORY_LAYOUT)
        best = self.optimization_target(semi_norm)(xi_temp), self.xi_len
        for m in range(self.xi_len):
            xi_temp[m] = 1.0 - EFFECTIVELY_ZERO
            measure = self.optimization_target(semi_norm)(xi_temp), m
            if measure[0] < best[0]:
                best = measure
            xi_temp[m] = 0.0
        if best[1] < self.xi_len:
            xi_temp[best[1]] = 1.0 - EFFECTIVELY_ZERO
        self.xi = xi_temp

    def optimize(self, semi_norm: SemiNorm, allow_rotation: bool, N_search: int = 2048, N_optimize: int = 1, xi: Optional[NP.Array] = None, **kwargs):
        q = transpose(self.Theta_old)
        for self._m in range(self.parameters.m_max.df.iloc[0, 0]):
            self.Theta_old = transpose(q)
            if allow_rotation:
                self._optimize_one(semi_norm, N_search, N_optimize, xi, **kwargs)
            else:
                self._optimal_shuffle(semi_norm)
            q, r = qr(transpose(self.Theta), check_finite=False)
        self.Theta_old = transpose(q)
        self.parameter_update()
        self.X_update()

    def parameter_update(self):
        m_saved = self._m
        self._m = 0
        xi_temp = zeros(self.xi_len, dtype=float, order=self.MEMORY_LAYOUT)
        for m in reversed(range(len(xi_temp))):
            xi_temp[m] = 1.0
            self.xi = xi_temp
            self._S1[:, :, m+1] = self.D[:, :, 0] / self.D[:, :, -1]
            xi_temp[m] = 0.0
        self.xi = xi_temp
        self._S1[:, :, 0] = self.D[:, :, 0] / self.D[:, :, -1]
        self._m = m_saved
        self.parameters = self.write_frames(self.Parameters(self.parameters.m_max, self.Theta_old,
                                                            self.Tensor3asMatrix(self.D), self.Tensor3asMatrix(self.S),
                                                            self.Tensor3asMatrix(self._S1)))

    def X_update(self):
        column_headings = array(["u{:d}".format(i) for i in range(self.gb.M)])
        self.gb.X = einsum('MK, NK -> NM', self.Theta_old, self.gb.X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        test_X = einsum('MK, NK -> NM', self.Theta_old, self.gb.fold.test_X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        self.gb.fold.data.df.iloc[:, :self.gb.M] = self.gb.X
        new_headings = self.gb.fold.data.df.columns.levels[1].values
        new_headings[self.gb.fold.data.df.columns.labels[1][:self.gb.M]] = column_headings
        self.gb.fold.data.df.columns.set_levels(new_headings, level=1, inplace=True)
        self.gb.fold.data.write()
        self.gb.fold.test.df.iloc[:, :self.gb.M] = test_X
        new_headings = self.gb.fold.test.df.columns.levels[1].values
        new_headings[self.gb.fold.test.df.columns.labels[1][:self.gb.M]] = column_headings
        self.gb.fold.test.df.columns.set_levels(new_headings, level=1, inplace=True)
        self.gb.fold.test.write()

    def Tensor3asMatrix(self, DorS: NP.Tensor3) -> NP.Matrix:
        return reshape(DorS, (self.gb.L, self.gb.L * self.gb.M))

    def _validate_parameters(self):
        if self.parameters.m_max.df.iloc[0, 0] not in range(self.gb.M):
            raise ValueError("We require 0 <= m_max={0:d} < M={1:d}".format(self.parameters.m_max.df.iloc[0, 0], self.gb.M))
        if self.parameters.Theta.df.shape != (self.gb.M, self.gb.M):
            raise ValueError("parameters.Theta.shape={0} when it should be (M, M)={1}".
                             format(self.parameters.Theta.df.shape, (self.gb.M, self.gb.M)))

    def __init__(self, gb: GaussianBundle, m_max: int = -1, overwrite: bool = True):
        self._gb = gb
        m_max = m_max if m_max in range(self.gb.M) else self.gb.M - 1
        self._Theta_old = eye(self.gb.M, dtype=float, order=self.MEMORY_LAYOUT)
        self._xi = None
        self._D = -ones((self.gb.L, self.gb.L, self.gb.M), dtype=float, order=self.MEMORY_LAYOUT)
        self._S1 = -ones((self.gb.L, self.gb.L, self.gb.M), dtype=float, order=self.MEMORY_LAYOUT)
        super().__init__(self.dir,
                         self.Parameters(m_max, self._Theta_old, self.Tensor3asMatrix(self.D), self.Tensor3asMatrix(self.S),
                                         self.Tensor3asMatrix(self._S1)), overwrite)
        self._validate_parameters()
        self._lengthscale = self.gb.kernel.parameters.lengthscale[0, :]
        if self._lengthscale.shape != (self.gb.M,):
            self._lengthscale = full(self.gb.M, self._lengthscale[0], dtype=float, order=self.MEMORY_LAYOUT)
        self.T_diagonal, self.Sigma_diagonal, self.Phi_diagonal = self._calculate_diagonals()
        self._T_pre, self._T_pre_cross, self._F, self._F1F = self._calculate_T_pre_cross_F_F1F()
        for self._m in reversed(range(self.gb.M)):
            self.Theta_old = self._Theta_old
        self.parameter_update()


class ROM(Model):

    Parameters = namedtuple("Parameters", ['m_max', 'D', 'S', 'S1', 'lengthscale', 'log_likelihood'])
    PARAMETER_DEFAULTS = Parameters(*(DataFrame(),) * 6)

    OPTIMIZED_GB_EXT = ".optimized"
    NAME = "ROM"

    @property
    def dir(self) -> Path:
        return self._gb.fold.dir / self.NAME

    @property
    def semi_norm(self) -> SemiNorm:
        return self._semi_norm

    def test(self) -> Frame:
        return self._sobol.gb.test()

    def optimize(self, allow_rotation: bool, iterations: int, guess_identity_after_iteration: int = -1,
                 N_search: int = 2048, N_optimize: int = 1, xi: Optional[NP.Array] = None, reuse_default_gb_parameters: bool = True, **kwargs):

        def _guess_lengthscale() -> NP.Matrix:
            return self.parameters.lengthscale.df.values * 0.5 * self._gb.M * (self._gb.M - arange(self._gb.M, dtype=float)) ** (-1)

        if iterations < 1 or not allow_rotation:
            if not iterations <= 1:
                warn("Your ROM optimization does not allow_rotation so iterations is set to 1, instead of {0:d}.".format(iterations), UserWarning)
            iterations = 1
        Theta = self._sobol.Theta_old
        if guess_identity_after_iteration < 0:
            guess_identity_after_iteration = iterations
        for i in range(iterations):
            if i < guess_identity_after_iteration:
                self._sobol.optimize(self._semi_norm, allow_rotation, N_search, N_optimize, xi, **kwargs)
            else:
                self._sobol.optimize(self._semi_norm, allow_rotation, 0, 1,
                                     xi=zeros(self._gb.M, dtype=float, order=self._sobol.MEMORY_LAYOUT), **kwargs)
            Theta = einsum('MK, KL -> ML', self._sobol.Theta_old, Theta)
            if reuse_default_gb_parameters:
                kernel_params = self._default_kernel_parameters
                params = self._default_parameters._replace(kernel=[[kernel_params]])
            else:
                kernel_params = self._gb.kernel.parameters
                if not kernel_params.ard:
                    lengthscale = _guess_lengthscale()
                    kernel_params = kernel_params._replace(lengthscale=lengthscale, ard=True)
                params = self._gb.parameters._replace(kernel=[[kernel_params]])
            self._gb = type(self._gb)(self._gb.fold, "{0}.{1:d}".format(self._gb.dir.stem, i + 1), params, True)
            self._gb.optimize()
            self._sobol = type(self._sobol)(self._gb)
            self.parameters = self.write_frames(self.Parameters(
                self.parameters.m_max,
                self.parameters.D.df.append(DataFrame(atleast_2d(self._semi_norm.value(self._sobol.D))), ignore_index=True),
                self.parameters.S.df.append(DataFrame(atleast_2d(self._semi_norm.value(self._sobol.S))), ignore_index=True),
                self.parameters.S1.df.append(DataFrame(atleast_2d(self._semi_norm.value(self._sobol.S1))), ignore_index=True),
                self.parameters.lengthscale.df.append(DataFrame(atleast_2d(self._sobol.lengthscale)), ignore_index=True),
                self.parameters.log_likelihood.df.append(self._gb.parameters.log_likelihood.df, ignore_index=True)))
        self._gb = type(self._gb)(self._gb.fold, self._gb.dir.stem + self.OPTIMIZED_GB_EXT,
                                  self._gb.parameters._replace(kernel=[[self._gb.kernel.parameters]]), True, False)
        self._sobol = type(self._sobol)(self._gb)
        frame = self._sobol.parameters.Theta
        frame.df.iloc[:] = Theta
        frame.write()

    def __init__(self, sobol: Sobol, semi_norm: SemiNorm, m_max: int = -1, overwrite=True):
        self._sobol = sobol
        self._gb = sobol.gb
        self._semi_norm = semi_norm
        self._default_kernel_parameters = self._gb.kernel.parameters
        self._default_parameters = self._gb.parameters
        self._gb.optimize()
        self._sobol = type(self._sobol)(self._gb)
        parameters = self.Parameters(m_max=m_max if m_max in range(self._gb.M) else self._gb.M - 1,
                                     D=self._semi_norm.value(self._sobol.D),
                                     S=self._semi_norm.value(self._sobol.S),
                                     S1=self._semi_norm.value(self._sobol.S1),
                                     lengthscale=self._sobol.lengthscale,
                                     log_likelihood=self._gb.parameters.log_likelihood.df.copy(deep=True))
        super().__init__(self.dir, parameters, overwrite)
        if overwrite:
            shutil.copy2(self._gb.fold.data_csv, self.dir)
            shutil.copy2(self._gb.fold.test_csv, self.dir)
