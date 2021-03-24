# BSD 3-Clause License
#
# Copyright (c) 2019-2021, Robert A. Milton
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

"""Contains base classes for various models.

Because implementation may involve parallelization, these classes should only contain pre-processing and post-processing.

"""

import shutil
from abc import ABC, abstractmethod
from enum import IntEnum, auto
from pathlib import Path
from warnings import warn
from numpy import atleast_2d, sqrt, einsum, exp, prod, eye, append, transpose, zeros, delete, diag, reshape, full, ones, empty, \
    arange, sum, concatenate, copyto, sign
from pandas import DataFrame, MultiIndex, concat
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve, qr
import json
from romcomma.data import Frame, Fold
from romcomma.typing_ import PathLike, Optional, NamedTuple, NP, Tuple, Type, Callable, Union, Any, List, Dict
from copy import deepcopy
from romcomma import distribution


class Model(ABC):
    """ Abstract base class for any model. This base class implements generic file storage and parameter handling.
    The latter is dealt with by each subclass overriding the ``Model.Parameters`` type with its own ``NamedTuple``
    defining the parameter set it takes.

    ``model.parameters`` is a ``NamedTuple`` of NP.Matrices.
    If ``model.with_frames``, each parameter is backed by a csv file, otherwise no file operations are involved.

    In case ``model.parameters is None``, ``model.parameters`` is read from ``model.dir``.
    """

    CSV_PARAMETERS = {'header': [0]}

    """ Required overrides."""

    class Parameters(NamedTuple):
        OVERRIDE_THIS: NP.Matrix
    DEFAULT_PARAMETERS: Parameters = "OVERRIDE_THIS"
    DEFAULT_OPTIMIZER_OPTIONS: Dict[str, Any] = {"OVERRIDE": "THIS"}
    MEMORY_LAYOUT: str = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-All-input-is-Fortran-layout)."

    """ End of required overrides."""

    @staticmethod
    def rmdir(dir_: Union[str, Path], ignore_errors: bool = True):
        """ Remove a directory tree, using shutil.

        Args:
            dir_: Root of the tree to remove.
            ignore_errors: Boolean.
        """
        shutil.rmtree(dir_, ignore_errors=ignore_errors)

    @staticmethod
    def copy(src_dir: Union[str, Path], dst_dir: Union[str, Path], ignore_errors: bool = True):
        """ Copy a directory tree, using shutil.

        Args:
            src_dir: Source root of the tree to copy.
            dst_dir: Destination root.
            ignore_errors: Boolean
        """
        shutil.rmtree(dst_dir, ignore_errors=ignore_errors)
        shutil.copytree(src=src_dir, dst=dst_dir)

    @property
    def dir(self) -> Path:
        """ The model directory."""
        return self._dir

    @property
    def with_frames(self) -> bool:
        """ Whether the Model has csv files backing its parameters."""
        return bool(self._dir.parts)

    @property
    def parameters(self) -> Parameters:
        """ Sets or gets the model parameters, as a NamedTuple of Matrices."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: Parameters):
        """ Sets or gets the model parameters, as a NamedTuple of Matrices."""
        self._parameters = value
        self.calculate()

    @property
    def parameters_csv(self) -> Parameters:
        """ A Model.Parameters NamedTuple of the csv files backing the model.parameters."""
        return self._parameters_csv

    def read_parameters(self):
        """ Read model.parameters from their csv files."""
        assert self.with_frames
        self._parameters = self.Parameters(*(Frame(self._parameters_csv[i], **self.CSV_PARAMETERS).df.values for i, p in enumerate(self.DEFAULT_PARAMETERS)))

    def write_parameters(self, parameters: Parameters) -> Parameters:
        """ Write model.parameters to their csv files.

        Args:
            parameters: The NamedTuple to be the new value for self.parameters.
        Returns: The NamedTuple written to csv. Essentially self.parameters, but with Frames in place of Matrices.
        """
        assert self.with_frames
        self._parameters = self.Parameters(*(atleast_2d(p) for p in parameters))
        return self.Parameters(*tuple(Frame(self._parameters_csv[i], DataFrame(p)) for i, p in enumerate(self._parameters)))

    @property
    def optimizer_options_json(self) -> Path:
        return self._dir / "optimizer_options.json"

    def _read_optimizer_options(self) -> dict:
        # noinspection PyTypeChecker
        with open(self.optimizer_options_json, mode='r') as file:
            return json.load(file)

    def _write_optimizer_options(self, optimizer_options: Dict):
        # noinspection PyTypeChecker
        with open(self.optimizer_options_json, mode='w') as file:
            json.dump(optimizer_options, file, indent=8)

    @abstractmethod
    def calculate(self):
        """ Calculate the Model. Do not call super().calculate, this interface only contains suggestions for implementation."""
        return
        self._test = None   # Remember to reset any test results.

    @abstractmethod
    def optimize(self, **kwargs):
        """ Optimize the model parameters. Do not call super().optimize, this interface only contains suggestions for implementation.

        Args:
            kwargs: Dict of implementation-dependent optimizer options.
        """
        return
        if kwargs is None:
            kwargs = self._read_optimizer_options() if self.optimizer_options_json.exists() else self.DEFAULT_OPTIMIZER_OPTIONS
        # OPTIMIZE!!!!!
        self._write_optimizer_options(kwargs)
        self.write_parameters(parameters=self.DEFAULT_PARAMETERS)   # Remember to write optimization results.
        self._test = None   # Remember to reset any test results.

    @abstractmethod
    def __init__(self, dir_: PathLike, parameters: Optional[Parameters] = None):
        """ Model constructor, to be called by all subclasses as a matter of priority.

        Args:
            dir_: The model file location. If and only if this is empty, then model.with_frames = False.
            parameters: The model.parameters, an Optional NamedTuple of NP.Matrices.
                If None then model.parameters are read from dir_, otherwise they are written to dir_, provided model.with_frames.
        """
        self._dir = Path(dir_)
        self._parameters_csv = self.Parameters(*((self._dir / field).with_suffix(".csv") for field in self.DEFAULT_PARAMETERS._fields))
        if parameters is None:
            self.read_parameters()
        else:
            self._dir.mkdir(mode=0o777, parents=True, exist_ok=True)
            self.write_parameters(parameters)
        self._test = None


# noinspection PyPep8Naming
class Kernel(Model):
    """ Abstract interface to a Kernel. Essentially this is the code contract with the GP interface.

    The kernel takes two input design matrices ``X0`` and ``X1``.

    If these are populated then ``not kernel.with_frames``, for efficiency.
    In this case, ``kernel.parameters.lengthscale.shape[1] &gt= kernel.M = X0.shape[1] = X1.shape[1]``.

    If ``X0 is None is X1`` then ``kernel.with_frames`` and the kernel is used purely for recording parameters.
    """

    """ Required overrides."""

    class Parameters(NamedTuple):
        OVERRIDE_THIS: NP.Matrix
    DEFAULT_PARAMETERS = "OVERRIDE_THIS"
    MEMORY_LAYOUT = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-All-input-is-Fortran-layout)."

    """ End of required overrides."""

    @staticmethod
    def TypeFromParameters(parameters: Parameters) -> Type['Kernel']:
        """ Recognize the Type of a Kernel from its Parameters.

        Args:
            parameters: A Kernel.Parameters array to recognize.
        Returns:
            The type of Kernel that parameters defines.
        """
        for kernel_type in Kernel.__subclasses__():
            if isinstance(parameters, kernel_type.Parameters):
                return kernel_type
        raise TypeError("Kernel Parameters array of unrecognizable type.")

    @classmethod
    def TypeIdentifier(cls) -> str:
        """ Returns the type of this Kernel object or class as "__module__.Kernel.__name__"."""
        return cls.__module__.split('.')[-1] + "." + cls.__name__

    @classmethod
    def TypeFromIdentifier(cls, TypeIdentifier: str) -> Type['Kernel']:
        """ Convert a TypeIdentifier to a Kernel Type.

        Args:
            TypeIdentifier: A string generated by Kernel.TypeIdentifier().
        Returns:
            The type of Kernel that _TypeIdentifier specifies.
        """
        for KernelType in cls.__subclasses__():
            if KernelType.TypeIdentifier() == TypeIdentifier:
                return KernelType
        raise TypeError("Kernel.TypeIdentifier() of unrecognizable type.")

    @property
    def X0(self) -> int:
        """ An (N0,M) Design (feature) Matrix containing the first argument to the kernel function."""
        return self._X0

    @property
    def N0(self) -> int:
        """ The number of datapoints (rows) in the first argument to the kernel function."""
        return self._N0

    @property
    def X1(self) -> int:
        """ An (N1,M) Design (feature) Matrix containing the second argument to the kernel function."""
        return self._X1

    @property
    def N1(self) -> int:
        """ The number of datapoints (rows) in the second argument to the kernel function."""
        return self._N1

    @property
    def M(self) -> int:
        """ The number of columns in the arguments to the kernel function. Must be the same for both arguments."""
        return self._M

    @property
    def matrix(self) -> NP.Matrix:
        """ The (N0,N1) kernel Matrix K(X0,X1)."""
        return self._matrix

    def optimize(self, options: Dict = Model.DEFAULT_OPTIMIZER_OPTIONS):
        """ Empty function, required by interface. Do not use.

        Args:
            kwrgsA Dict of implementation-dependent optimizer options, following the format of Model.DEFAULT_OPTIMIZER_OPTIONS.
        """

    # noinspection PyUnusedLocal
    @abstractmethod
    def __init__(self, X0: Optional[NP.Matrix], X1: Optional[NP.Matrix], dir_: PathLike, parameters: Optional[Parameters] = None):
        """ Construct a Kernel.

        Args:
            X0: An (N0,M) Design (feature) Matrix. Use None if and only if kernel is only for recording parameters.
            X1: An (N1,M) Design (feature) Matrix. Use None if and only if kernel is only for recording parameters.
            dir_: The kernel file location.
            parameters: The kernel parameters. If None these are read from dir_.
        """
        super().__init__(dir_, parameters)
        self._matrix = None
        self._X0 = X0
        self._X1 = X1
        if self._X0 is not None and self._X1 is not None:
            assert X0.shape[1] == X1.shape[1], "X0 and X1 have differing numbers of columns."
            self._N0, self._N1, self._M = X0.shape[0], *X1.shape
        else:
            self._N0, self._N1, self._M = 0, 0, 0


# noinspection PyPep8Naming
class GP(Model):
    """ Interface to a Gaussian Process."""

    """ Required overrides."""

    MEMORY_LAYOUT = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-All-input-is-Fortran-layout)."

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
    DEFAULT_PARAMETERS = Parameters(kernel=None, e_floor=atleast_2d(1E-12), f=atleast_2d(0.9), e=atleast_2d(0.1),
                                    log_likelihood=atleast_2d(None))

    DEFAULT_OPTIMIZER_OPTIONS = {"OVERRIDE": "THIS"}

    KERNEL_NAME = "kernel"

    """ End of required overrides."""

    @property
    def fold(self) -> Fold:
        """ The parent fold """
        return self._fold

    @property
    def test_csv(self) -> Path:
        return self._dir / "__test__.csv"

    @property
    def N(self) -> int:
        """ The number of input rows = The number of output rows  = datapoints in the training set."""
        return self._N

    @property
    def M(self) -> int:
        """ The number of input columns."""
        return self._M

    @property
    def L(self) -> int:
        """ The number of output columns."""
        return self._L

    @property
    def X(self) -> NP.Matrix:
        """ The input X, as an (N,M) design Matrix."""
        return self._X

    @property
    def Y(self) -> NP.Vector:
        """ The output Y, as an (N,L) NP.Matrix."""
        return self._Y

    @property
    @abstractmethod
    def log_likelihood(self) -> float:
        """ The log marginal likelihood of the training data given the GP parameters."""

    @property
    def kernel(self) -> Kernel:
        """ The GP Kernel. """
        return self._kernel

    @abstractmethod
    def optimize(self, options: Dict = DEFAULT_OPTIMIZER_OPTIONS):
        """ Empty function, required by interface. Do not use.

        Args:
            kwrgsA Dict of implementation-dependent optimizer options, following the format of GP.DEFAULT_OPTIMIZER_OPTIONS.
        """

    @abstractmethod
    def predict(self, X: NP.Matrix, Y_instead_of_F: bool = True) -> Tuple[NP.Matrix, NP.Matrix, NP.Tensor3]:
        """ Predicts the response to input X.

        Args:
            X: An (N,M) design Matrix of inputs.
            Y_instead_of_F: True to include noise e in the result covariance.
        Returns: The distribution of Y or f, as a triplet (mean (N, L) Matrix, std (N, L) Matrix, covariance (N, L, L) Tensor3).
        """

    def test(self, full_cov: bool = False) -> Frame:
        """ Tests the GP on the test data in GP.fold.test_csv.

        Args:
            full_cov: Whether to return the full output covariance (N,L,L) Tensor3, or just the output variance (N,L) Matrix.
        Returns: The test results as a Frame backed by GP.test_result_csv.
        """
        if self._test is None:
            self._test = Frame(self.test_csv, self._fold.test.df)
            Y_heading = self._fold.meta['data']['Y_heading']
            result = self.predict(self._fold.test_X.values)
            predictive_mean = (self._test.df.loc[:, [Y_heading]].copy().rename(columns={Y_heading: "Predictive Mean"}, level=0))
            predictive_mean.iloc[:] = result[0]
            predictive_std = (self._test.df.loc[:, [Y_heading]].copy().rename(columns={Y_heading: "Predictive Std"}, level=0))
            predictive_std.iloc[:] = result[1]
            self._test.df = self._test.df.join([predictive_mean, predictive_std])
            if full_cov and self._L > 1:
                output_headings = self._fold.test_Y.columns
                for l in range(self._L):
                    predictive_std = (self._test.df.loc[:, [Y_heading]].copy().rename(columns={Y_heading: output_headings[l]}, level=0))
                    predictive_std.iloc[:] = result[2][:, :, l]
                    self._test.df = self._test.df.join(predictive_std)
            self._test.write()
        return self._test

    @property
    @abstractmethod
    def inv_prior_Y_Y(self) -> NP.Matrix:
        """ The (N,L) Matrix (f K(X,X) + e I)^(-1) Y."""

    @property
    def Yt(self) -> NP.Matrix:
        """ An (L, N) Matrix, known in the literature as Y_tilde. """
        return einsum('LK, NK -> LN', self.parameters.f, self.inv_prior_Y_Y, dtype=float, order=self.MEMORY_LAYOUT, optimize=True)

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

    # noinspection PyUnresolvedReferences
    def _validate_parameters(self):
        """ Generic validation.

        Raises:
            IndexError: if parameters.kernel and parameters.e_floor are not shaped (1,1).
            IndexError: unless parameters.f.shape == parameters.e == (1,1) or (L,L).
        """
        if self.parameters.kernel.shape != (1, 1):
            raise IndexError("GaussianProcess.parameters.kernel.shape must be (1,1), not {0}.".format(self.parameters.kernel.shape))
        if self.parameters.e_floor.shape != (1, 1):
            raise IndexError("GaussianProcess.parameters.e_floor.shape must be (1,1), not {0}.".format(self.parameters.e_floor.shape))
        if not (self.parameters.f.shape == self.parameters.e.shape):
            raise ValueError("GaussianProcess.parameters requires f and e parameters to be the same shape.")
        if self.parameters.e.shape not in ((1, 1), (self._L, self._L)):
            raise IndexError("GaussianProcess.parameters.e.shape must be (1,1) or (L,L), not {0}.".format(self.parameters.e.shape))

    @abstractmethod
    def __init__(self, fold: Fold, name: str, parameters: Optional[Parameters] = None):
        """ GP Constructor. Calls model.__Init__ to setup parameters, then checks dimensions.

        Args:
            fold: The Fold housing this GaussianProcess.
            name: The name of this GaussianProcess.
            parameters: The model parameters. If None these are read from fold/name, otherwise they are written to fold/name.
                parameters.kernel, if provided, must be a Kernel.Parameters NamedTuple (not a numpy array!) storing the desired kernel parameters.
        """
        self._fold = fold
        self._dir = fold.dir / name
        self._test = None
        self._X = self._fold.X.values.copy(order=self.MEMORY_LAYOUT)
        self._Y = self._fold.Y.values.copy(order=self.MEMORY_LAYOUT)
        self._N, self._M, self._L = self._fold.N, self._fold.M, self._fold.L
        if parameters is None:
            super().__init__(self._dir, parameters)
            KernelType = Kernel.TypeFromIdentifier(self._parameters.kernel[0, 0])
            self._kernel = KernelType(X0=self.X, X1=self.X, dir_=self._dir/self.KERNEL_NAME, parameters=None)
        else:
            KernelType = Kernel.TypeFromParameters(parameters.kernel)
            self._kernel = KernelType(X0=self.X, X1=self.X, dir_=self._dir/self.KERNEL_NAME, parameters=parameters.kernel)
            parameters = parameters._replace(kernel=atleast_2d(KernelType.TypeIdentifier()))
            super().__init__(self._dir, parameters)
        self._validate_parameters()


# noinspection PyPep8Naming
class Sobol(Model):
    """ Interface to a Sobol' Index Calculator and Optimizer.
    
    Internal quantities are called variant if they depend on Theta, invariant otherwise. 
    Invariants are calculated in the constructor. Variants are calculated in Theta.setter."""

    """ Required overrides."""

    MEMORY_LAYOUT = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-All-input-is-Fortran-layout)."

    Parameters = NamedTuple("Parameters", [('Theta', NP.Matrix), ('D', NP.Matrix), ('S1', NP.Matrix), ('S', NP.Matrix), ('ST', NP.Matrix)])
    """ 
        **Theta** -- The (Mu, M) rotation matrix ``U = X Theta.T`` (in design matrix terms), so u = Theta x (in column vector terms).

        **D** -- An (L L, M) Matrix of cumulative conditional variances D[l,k,m] = S[l,k,m] D[l,k,M].

        **S1** -- An (L L, M) Matrix of Sobol' first order indices.

        **S** -- An (L L, M) Matrix of Sobol' closed indices.

        **ST** -- An (L L, M) Matrix of Sobol' total indices.
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

    """ End of required overrides."""

    @property
    def gp(self):
        """ The underlying GP."""
        return self._gp

    @property
    def D(self) -> NP.Tensor3:
        """ An (L, L, Mx) Tensor3 of conditional variances."""
        return self._D

    @property
    def S(self) -> NP.Tensor3:
        """ An (L, L, M) Tensor3 of Closed (cumulative) Sobol' indices."""
        return self._D / self._D[:, :, -1]

    @property
    def S1(self) -> NP.Tensor3:
        """ An (L, L, Mx) Tensor3 of first order (main effect) Sobol' indices."""
        return self._S1

    @property
    def ST(self) -> NP.Tensor3:
        """ An (L, L, Mx) Tensor3 of Total Sobol' indices."""
        return self._ST

    @property
    def lengthscale(self):
        """ An (Mx,) Array of ARD lengthscales."""
        return self._lengthscale

    def Tensor3AsMatrix(self, DorS: NP.Tensor3) -> NP.Matrix:
        return reshape(DorS, (self._L * self._L, self._M))

    # @property
    # def Theta_old(self) -> NP.Matrix:
    #     """ Sets or gets the (M, M) rotation Matrix, prior to updates by xi. Setting automatically updates Theta, triggering Sobol' recalculation."""
    #     return self._Theta_old
    #
    # @Theta_old.setter
    # def Theta_old(self, value: NP.Matrix):
    #     assert value.shape == (self._M, self._M)
    #     self._Theta_old = value
    #     self.Theta = self.Theta_old[:self._m + 1, :].copy(order=self.MEMORY_LAYOUT)
    #
    # @property
    # def Theta(self) -> NP.Matrix:
    #     """ Sets or gets the (m+1, Mx) rotation Matrix which has been updated by xi. Setting triggers Sobol' recalculation."""
    #     return self._Theta
    #
    # # noinspection PyAttributeOutsideInit
    # @Theta.setter
    # def Theta(self, value: NP.Matrix):
    #     """ Complete calculation of variants and Sobol' indices (conditional variances _D actually) is found here and here only."""
    #
    #     assert value.shape == (self._m + 1, self._M)
    #     self._Theta = value
    #
    #     """ Calculate variants related to Sigma. """
    #     self._Sigma_partial = einsum('M, kM -> Mk', self._Sigma_diagonal, self.Theta, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     Sigma = einsum('mM, Mk -> mk', self.Theta, self._Sigma_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self._Sigma_cho = cho_factor(Sigma, lower=False, overwrite_a=False, check_finite=False)
    #     Sigma_cho_det = prod(diag(self._Sigma_cho[0]))
    #     self._2I_minus_Sigma_partial = einsum('M, kM -> Mk', 2 - self._Sigma_diagonal, self.Theta, optimize=True, dtype=float,
    #                                           order=self.MEMORY_LAYOUT)
    #     _2I_minus_Sigma = einsum('mM, Mk -> mk', self.Theta, self._2I_minus_Sigma_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self._2I_minus_Sigma_cho = cho_factor(_2I_minus_Sigma, lower=False, overwrite_a=False, check_finite=False)
    #     _2I_minus_Sigma_cho_det = prod(diag(self._2I_minus_Sigma_cho[0]))
    #     self._inv_Sigma_Theta = cho_solve(self._Sigma_cho, self.Theta, overwrite_b=False, check_finite=False)
    #     T_inv_Sigma_T = atleast_2d(einsum('NM, mM, mK, NK -> N', self._FBold, self.Theta, self._inv_Sigma_Theta, self._FBold,
    #                                       optimize=True, dtype=float, order=self.MEMORY_LAYOUT))
    #     self._D_const = (_2I_minus_Sigma_cho_det * Sigma_cho_det) ** (-1)
    #
    #     """ Calculate variants related to Phi. """
    #     self._Phi_partial = einsum('M, kM -> Mk', self._Psi_diagonal, self.Theta, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     Phi = einsum('mM, Mk -> mk', self.Theta, self._Phi_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self._Phi_cho = cho_factor(Phi, lower=False, overwrite_a=False, check_finite=False)
    #     self._inv_Phi_inv_Sigma_Theta = cho_solve(self._Phi_cho, self._inv_Sigma_Theta, overwrite_b=False, check_finite=False)
    #     T_inv_Phi_inv_Sigma_T = einsum('NOM, mM, mK, NOK -> NO', self._V_pre_outer_square, self.Theta, self._inv_Phi_inv_Sigma_Theta,
    #                                     self._V_pre_outer_square, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #
    #     """ Finally calculate conditional variances _D."""
    #     self._W = exp(-0.5 * (T_inv_Sigma_T + transpose(T_inv_Sigma_T) - T_inv_Phi_inv_Sigma_T))
    #     self._D_plus_Ft_1_Ft = self._D_const * einsum('LN, NO, KO -> LK', self._fBold_bar_0, self._W, self._fBold_bar_0,
    #                                                   optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self._D[:, :, self._m] = self._D_plus_Ft_1_Ft - self._f_bar_0_2
    #
    # @property
    # def xi(self) -> NP.Array:
    #     """ Sets or gets the (Mx-m-1) Array which is the row m update to Theta. Setting updates Theta, so triggers Sobol' recalculation."""
    #     return self._xi
    #
    # @xi.setter
    # def xi(self, value: NP.Array):
    #     assert value.shape[0] == self._xi_len
    #     norm = 1 - einsum('m, m -> ', value, value, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     if norm < EFFECTIVELY_ZERO:
    #         value *= sqrt((1-EFFECTIVELY_ZERO)/(1-norm))
    #         norm = EFFECTIVELY_ZERO
    #     self._xi = append(sqrt(norm), value)
    #     self.Theta[self._m, :] = einsum('k, kM -> M', self._xi, self.Theta_old[self._m:, :],
    #                                    optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self.Theta = self.Theta[:self._m + 1, :]

    def optimize(self, **kwargs):
        """ Optimize ``Theta`` to maximize ``semi_norm(D[:,:,m])`` for ``m=0,1,..(Mu-1)``.

        Args:
            options: A Dict similar to (and documented in) Sobol.DEFAULT_OPTIMIZER_OPTIONS.
        Raises:
            TypeError: Unless options['SemiNorm'] is a Sobol.SemiNorm or a Dict.
            UserWarning: If L for the SemiNorm must be changed to match self._L.
        """
        options = deepcopy(kwargs)
        if options is None:
            options = self._read_optimizer_options() if self.optimizer_options_json.exists() else self.DEFAULT_OPTIMIZER_OPTIONS
        semi_norm = Sobol.SemiNorm.from_meta(options['semi_norm'])
        if semi_norm.meta['L'] != self._L:
            warn("I am changing Sobol.semi_norm.meta['L'] from {0:d} to {1:d} = Sobol.gp.L.".format(semi_norm.meta['L'], self._L))
            semi_norm.meta['L'] = self._L
            semi_norm = Sobol.SemiNorm.from_meta(semi_norm.meta)
        options['semi_norm'] = semi_norm.meta
        self._write_optimizer_options(options)
        #
        # def _objective_value(xi: NP.Array) -> float:
        #     """ Maps ``xi`` to the optimization objective (a conditional variance ``D'').
        #
        #     Args:
        #         xi: The Theta row update.
        #     Returns: The scalar (float) -semi_norm(D[:, :, m]).
        #     """
        #     self.xi = xi
        #     return -semi_norm.value(self._D[:, :, self._m])
        #
        # def _objective_jacobian(xi: NP.Array) -> NP.Array:
        #     """ Maps ``xi`` to an ``(Mx-m,) Array``, the optimization objective jacobian.
        #
        #     Args:
        #         xi: The Theta row update.
        #     Returns: The (Mx-m,) jacobian Array -d(semi_norm(D[:, :, m])) / (d xi).
        #     """
        #     self.xi = xi
        #     return -einsum('LK, LKj -> j', semi_norm.derivative(self.D[:, :, self._m]), self.D_jacobian,
        #                    optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        #
        # self._objective_value, self._objective_jacobian = _objective_value, _objective_jacobian
        #
        # if options['N_exploit'] >= 1:
        #     q = None
        #     for self.m in range(self._Mu):
        #         if q is not None:
        #             self.Theta_old = transpose(q)
        #         self.xi = self._optimal_rotating_xi(options['N_explore'], options['N_exploit'], options['options'])
        #         q, r = qr(transpose(self.Theta), check_finite=False)
        #         sign_correction = sign(diag(r))
        #         q *= concatenate((sign_correction, ones(self._M - len(sign_correction), dtype=int)))
        #     self.Theta_old = transpose(q)
        # q = None
        # for self.m in range(self._Mu):
        #     if q is not None:
        #         self.Theta_old = transpose(q)
        #     self.xi = self._optimal_reordering_xi()
        #     q, r = qr(transpose(self.Theta), check_finite=False)
        #     # sign_correction = sign(diag(r))
        #     # q *= concatenate((sign_correction, ones(self._M - len(sign_correction), dtype=int)))
        # self.Theta_old = transpose(q)
        # self.write_parameters(self.Parameters(Mu=self.Mu, Theta=self._Theta_old, D=self.Tensor3AsMatrix(self._D), S1=None,
        #                                       S=self.Tensor3AsMatrix(self.S)))
        # self.replace_X_with_U()

    # def write_parameters(self, parameters: Parameters) -> Parameters:
    #     """ Calculate the main Sobol' indices _S1, then write model.parameters to their csv files.
    #
    #     Args:
    #         parameters: The NamedTuple to be the new value for self.parameters.
    #     Returns: The NamedTuple written to csv. Essentially self.parameters, but with Frames in place of Matrices.
    #     """
    #     if self._m is not None:
    #         m_saved = self._m
    #         self.m = 0
    #         xi_temp = zeros(self._xi_len, dtype=float, order=self.MEMORY_LAYOUT)
    #         for m in reversed(range(len(xi_temp))):
    #             xi_temp[m] = 1.0
    #             self.xi = xi_temp
    #             self._S1[:, :, m+1] = self.D[:, :, 0] / self.D[:, :, -1]
    #             xi_temp[m] = 0.0
    #         self.xi = xi_temp
    #         self._S1[:, :, 0] = self.D[:, :, 0] / self.D[:, :, -1]
    #         self.m = m_saved
    #     return super().write_parameters(parameters._replace(S1=self.Tensor3AsMatrix(self._S1)))

    def replace_X_with_U(self):
        """ Replace X with its rotated/reordered version U."""
        column_headings = MultiIndex.from_product(((self._gp.fold.meta['data']['X_heading'],), ("u{:d}".format(i) for i in range(self.Mu))))
        X = DataFrame(einsum('MK, NK -> NM', self.Theta_old, self._gp.X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT),
                      columns=column_headings, index=self._gp.fold.X.index)
        test_X = DataFrame(einsum('MK, NK -> NM', self.Theta_old, self._gp.fold.test_X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT),
                      columns=column_headings, index=self._gp.fold.test_X.index)
        self._gp.fold.data.df = concat((X, self._gp.fold.data.df[[self._gp.fold.meta['data']['Y_heading']]].copy(deep=True)), axis='columns')
        self._gp.fold.data.write()
        self._gp.fold.test.df = concat((test_X, self._gp.fold.test.df[[self._gp.fold.meta['data']['Y_heading']]].copy(deep=True)), axis='columns')
        self._gp.fold.test.write()
        self._gp.fold.meta_data_update()

    def reorder_data_columns(self, reordering: NP.Array):
        """ Reorder the columns in self._gp.data

        Args:
            reordering: An array of column indices specifying the reordering
        """
        columns = self._gp.fold.data.df.columns
        new_columns = columns.levels[1].to_numpy()
        new_columns[:reordering.shape[0]] = new_columns[reordering]
        columns = columns.reindex(new_columns, level=1)
        self._gp.fold.data.df.reindex(columns=columns, copy=False)
        self._gp.fold.data.write()
        self._gp.fold.test.df.reindex(columns=columns, copy=False)
        self._gp.fold.test.write()

    # def exploratory_xi(self, N_explore: int) -> NP.Matrix:
    #     """ Generate a matrix of xi's to explore.
    #
    #     Args:
    #         N_explore: The maximum number of xi's to explore. If N_explore &le 1 the zero vector is returned.
    #             This is tantamount to taking Theta = I (the identity matrix).
    #         xi_len: The length of each xi (row) Array.
    #     Returns: An (N_explore, xi_len) Matrix where 1 &le N_explore &le max(N_explore,1).
    #     """
    #     assert self._xi_len > 0
    #     if N_explore <= 1:
    #         return zeros((1, self._xi_len))
    #     else:
    #         N_explore = round(N_explore**((self._xi_len + 1) / self._M))
    #         dist = distribution.Multivariate.Independent(self._xi_len + 1, distribution.Univariate('uniform', loc=-1, scale=2))
    #         result = dist.sample(N_explore, distribution.SampleDesign.LATIN_HYPERCUBE)
    #         norm = sqrt(sum(result ** 2, axis=1).reshape((N_explore, 1)))
    #         return result[:, 1:] / norm
    #
    #     """
    #     if N_explore <= 1:
    #         return zeros((1, xi_len))
    #     elif N_explore < 3 ** xi_len:
    #         result = random.randint(3, size=(N_explore, xi_len)) - 1
    #     else:
    #         values = array([-1, 0, 1])
    #         # noinspection PyUnusedLocal
    #         values = [values.copy() for i in range(xi_len)]
    #         result = meshgrid(*values)
    #         result = [ravel(arr, order=cls.MEMORY_LAYOUT) for arr in result]
    #         result = concatenate(result, axis=0)
    #     return result * xi_len ** (-1 / 2)
    #     """
    #
    # def _optimal_rotating_xi(self, N_explore: int, N_exploit: int, options: Dict) -> NP.Array:
    #     """ Optimizes the ``Theta`` row update ``xi`` by allowing general rotation.
    #
    #     Args:
    #         N_explore: The number of random_sgn xi vectors to explore in search of the global optimum.
    #         N_exploit: The number of exploratory xi vectors to exploit (gradient descend) in search of the global optimum.
    #         options: A Dict of options passed directly to the underlying optimizer.
    #     Returns:
    #         The Array xi of euclidean length &le 1 which maximizes self.optimization_objective(xi).
    #     """
    #     explore = self.exploratory_xi(N_explore)
    #     best = [[0, explore[0]]] * N_exploit
    #     for xi in explore:
    #         objective_value = self._objective_value(xi)
    #         for i in range(N_exploit):
    #             if objective_value < best[i][0]:
    #                 for j in reversed(range(i + 1, N_exploit)):
    #                     best[j] = best[j - 1]
    #                 best[i] = [objective_value, xi]
    #                 break
    #     for record in best:
    #         result = optimize.minimize(self._objective_value, record[1], method='BFGS', jac=self._objective_jacobian, options=options)
    #         if result.fun < best[0][0]:
    #             best[0] = [result.fun, result.x]
    #     return best[0][1]
    #
    # def _optimal_reordering_xi(self) -> NP.Array:
    #     """ Optimizes the ``Theta`` row update ``xi`` by allowing re-ordering only.
    #
    #     Returns:
    #         The Array xi consisting of all zeros except at most one 1.0 which maximizes self.optimization_objective(xi).
    #     """
    #     xi = zeros(self._xi_len, dtype=float, order=self.MEMORY_LAYOUT)
    #     best = self._objective_value(xi), self._xi_len
    #     for m in range(self._xi_len):
    #         xi[m] = 1.0 - EFFECTIVELY_ZERO
    #         objective = self._objective_value(xi), m
    #         if objective[0] < best[0]:
    #             best = objective
    #         xi[m] = 0.0
    #     if best[1] < self._xi_len:
    #         xi[best[1]] = 1.0 - EFFECTIVELY_ZERO
    #     return xi
    #
    # # noinspection PyAttributeOutsideInit
    # @property
    # def D_jacobian(self) -> NP.Tensor3:
    #     """ Calculate the Jacobian d (``D[:, ;, m]``) / d (``xi``).
    #
    #     Returns:  The Tensor3(L, L, M-m) jacobian d(D[:, ;, m]) / d (xi).
    #     """
    #
    #     """ Calculate various jacobians."""
    #     Theta_jac = zeros((self._m + 1, self._M, self._xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
    #     Theta_jac[self._m, :, :] = transpose(self.Theta_old[self._m:self._M, :])
    #     Sigma_jac = einsum('mMj, Mk -> mkj', Theta_jac, self._Sigma_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     inv_Sigma_jac_Sigma = zeros((self._m + 1, self._m + 1, self._xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
    #     _2I_minus_Sigma_jac = einsum('mMj, Mk -> mkj', Theta_jac, self._2I_minus_Sigma_partial, optimize=True,
    #                                       dtype=float, order=self.MEMORY_LAYOUT)
    #     inv_2I_minus_Sigma_jac_2I_minus_Sigma = zeros((self._m + 1, self._m + 1, self._xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
    #     Phi_jac = einsum('mMj, Mk -> mkj', Theta_jac, self._Phi_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     inv_Phi_jac_Phi = zeros((self._m + 1, self._m + 1, self._xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
    #     inv_Phi_inv_Sigma_jac_Sigma = zeros((self._m + 1, self._m + 1, self._xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
    #     log_D_const_jac = zeros(self._xi_len + 1, dtype=float, order=self.MEMORY_LAYOUT)
    #     for j in range(self._xi_len + 1):
    #         inv_Sigma_jac_Sigma[:, :, j] = cho_solve(self._Sigma_cho, Sigma_jac[:, :, j], overwrite_b=False, check_finite=False)
    #         inv_2I_minus_Sigma_jac_2I_minus_Sigma[:, :, j] = cho_solve(self._2I_minus_Sigma_cho, _2I_minus_Sigma_jac[:, :, j],
    #                                                                          overwrite_b=False, check_finite=False)
    #         inv_Phi_jac_Phi[:, :, j] = cho_solve(self._Phi_cho, Phi_jac[:, :, j], overwrite_b=False, check_finite=False)
    #         inv_Phi_inv_Sigma_jac_Sigma[:, :, j] = cho_solve(self._Phi_cho, inv_Sigma_jac_Sigma[:, :, j], overwrite_b=False, check_finite=False)
    #         log_D_const_jac[j] = (sum(diag(inv_2I_minus_Sigma_jac_2I_minus_Sigma[:, :, j])) + sum(diag(inv_Sigma_jac_Sigma[:, :, j])))
    #
    #     """ Calculate self._V, a Tensor3(N, N, M-m,) known in the literature as V. """
    #     Sigma_factor_transpose = Theta_jac - einsum('kmj, kM -> mMj', inv_Sigma_jac_Sigma, self.Theta,
    #                                                       optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     Theta_inv_Sigma_Theta_jac = einsum('mMj, mK -> MKj', Sigma_factor_transpose, self._inv_Sigma_Theta,
    #                                        optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     T_inv_Sigma_T_jac = einsum('NM, MKj, NK -> Nj', self._FBold, Theta_inv_Sigma_Theta_jac, self._FBold,
    #                                optimize=True, dtype=float, order=self.MEMORY_LAYOUT).reshape((1, self._N, self._xi_len + 1),
    #                                                                                               order=self.MEMORY_LAYOUT)
    #     Phi_factor_transpose = Theta_jac - einsum('kmj, kM -> mMj', inv_Phi_jac_Phi, self.Theta,
    #                                               optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     Theta_inv_Phi_inv_Sigma_Theta_jac = einsum('mMj, mK -> MKj', Phi_factor_transpose, self._inv_Phi_inv_Sigma_Theta,
    #                                                optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     Theta_inv_Phi_inv_Sigma_Theta_jac -= einsum('kM, kmj, mK -> MKj', self.Theta, inv_Phi_inv_Sigma_jac_Sigma, self._inv_Sigma_Theta,
    #                                                 optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     T_inv_Phi_inv_Sigma_T_jac = einsum('NOM, MKj, NOK -> NOj',
    #                                        self._V_pre_outer_square, Theta_inv_Phi_inv_Sigma_Theta_jac, self._V_pre_outer_square,
    #                                        optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self._V = T_inv_Phi_inv_Sigma_T_jac - (T_inv_Sigma_T_jac + transpose(T_inv_Sigma_T_jac, (1, 0, 2)))
    #
    #     """ Calculate D_jacobian. """
    #     D_derivative = self._D_const * einsum('LN, NO, NOj, KO -> LKj', self._fBold_bar_0, self._W, self._V, self._fBold_bar_0,
    #                                           optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     D_derivative -= einsum('j, LK -> LKj', log_D_const_jac, self._D_plus_Ft_1_Ft, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     result = -self._xi[1:] / self._xi[0]
    #     result = einsum('LK, j -> LKj', D_derivative[:, :, 0], result, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     return result + D_derivative[:, :, 1:]

    def calculate(self):
        """ Calculate the Sobol' Indices S1 and S."""
        def _diagonal_matrices():
            """ Cache the diagonal parts of the key (Mx,Mx) matrices V, Sigma, Phi.
                _FBold_diagonal = (_lengthscale^(2) + I)^(-1)
                _Sigma_diagonal = (lengthscale^(-2) + I)^(-1)
                _Psi_diagonal = (2*lengthscale^(-2) + I) (lengthscale^(-2) + I)^(-1)
            All invariant.
            """
            self._FBold_diagonal = self._lengthscale ** 2
            self._Psi_diagonal = self._FBold_diagonal ** (-1)
            self._FBold_diagonal += 1.0
            self._Psi_diagonal += 1.0
            self._2Sigma_diagonal = 2 * (self._Psi_diagonal ** (-1))
            self._FBold_diagonal = self._FBold_diagonal ** (-1)
            self._2Psi_diagonal = (2 * self._Psi_diagonal - 1.0) * self._2Sigma_diagonal
        _diagonal_matrices()

        def _precursors_to_D():
            """ Cache invariant precursors to _D."""
            self._FBold = einsum('NM, M -> MN', self._gp.X, self._FBold_diagonal, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            self._FBold_outer_plus = self._FBold_outer_minus = self._FBold.reshape((self._M, 1, self._N))
            FBold_T = transpose(self._FBold_outer_minus, (0, 2, 1))
            self._FBold_outer_plus = self._FBold_outer_plus + FBold_T
            self._FBold_outer_minus = self._FBold_outer_minus - FBold_T
        _precursors_to_D()

        def _conditional_expectations():
            """ Cache invariant conditional expectations _fBold_bar_0 and _f_bar_0_2. """
            self._fBold_bar_0 = einsum('MN, NM -> N', self._FBold, self._gp.X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            self._fBold_bar_0 = sqrt(prod(self._2Sigma_diagonal) / (2**self._M)) * exp(-0.5 * self._fBold_bar_0)
            self._fBold_bar_0 = einsum('Ll, Nl, N -> LN', self._gp.parameters.f, self._gp.inv_prior_Y_Y, self._fBold_bar_0, optimize=True,
                                       dtype=float, order=self.MEMORY_LAYOUT)
            self._f_bar_0_2 = einsum('LN -> L', self._fBold_bar_0, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            self._f_bar_0_2 = einsum('L, l -> Ll', self._f_bar_0_2, self._f_bar_0_2, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        _conditional_expectations()

        def _DBold_DBold_full_DBold_full_determinant() -> Tuple[NP.Matrix, NP.Tensor3, NP.Array]:
            """ Calculate DBold and its expanded form DBold_full, with a determinant pre-factor. """
            DBold_full_determinant = (self._2Sigma_diagonal / self._2Psi_diagonal) ** (1/2)
            DBold_numerator_precision_halved = 0.5 * (self._2Sigma_diagonal ** (-1))
            DBold_denominator_precision_halved = 0.5 * (self._2Psi_diagonal ** (-1))
            DBold_full = einsum('MNO, M, MNO -> MNO', self._FBold_outer_minus, DBold_numerator_precision_halved, self._FBold_outer_minus,
                                  optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            DBold_full -= einsum('MNO, M, MNO -> MNO', self._FBold_outer_plus, DBold_denominator_precision_halved, self._FBold_outer_plus,
                                  optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            DBold = einsum('MNO -> NO', DBold_full, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            return ((prod(DBold_full_determinant) ** (-1)) * exp(-DBold), exp(DBold_full), DBold_full_determinant)
        DBold, DBold_full, DBold_full_determinant = _DBold_DBold_full_DBold_full_determinant()

        def sobol_and_reorder() -> NP.Array:
            """ Calculate all Sobol Indices, returning a reordering by decreasing ST_m """
            self._D[:, :, -1] = einsum('LN, NO, lO -> Ll', self._fBold_bar_0, DBold, self._fBold_bar_0,
                                            optimize=True, dtype=float, order=self.MEMORY_LAYOUT) - self._f_bar_0_2
            DBold_m_included = DBold.copy(order=self.MEMORY_LAYOUT)
            DBold_m_to_include = empty((self._N, self._N), dtype=float, order=self.MEMORY_LAYOUT)
            DBold_m = empty((self._N, self._N), dtype=float, order=self.MEMORY_LAYOUT)
            reordering = -ones(self._M, dtype=int, order=self.MEMORY_LAYOUT)
            D_m = empty((self._L, self._L), dtype=float, order=self.MEMORY_LAYOUT)
            S_m = empty((self._L, self._L), dtype=float, order=self.MEMORY_LAYOUT)
            for m_excluded in reversed(range(1, self._M)):
                max_semi_norm = (-1, -1.5)
                for m_to_exclude in reversed(range(self._M)):
                    if m_to_exclude not in reordering[m_excluded:]:
                        DBold_m[:] = DBold_m_included * DBold_full[m_to_exclude, :, :] * DBold_full_determinant[m_to_exclude]
                        einsum('LN, NO, lO -> Ll', self._fBold_bar_0, DBold_m, self._fBold_bar_0, out=D_m,
                               optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
                        D_m -= self._f_bar_0_2
                        S_m[:] = D_m / self._D[:, :, -1]
                        semi_norm = self._semi_norm.value(S_m)
                        if semi_norm > max_semi_norm[1]:
                            max_semi_norm = (m_to_exclude, semi_norm)
                            copyto(DBold_m_to_include, DBold_m)
                            self._D[:, :, m_excluded - 1] = D_m[:]
                reordering[m_excluded] = max_semi_norm[0]
                self._S[:, :, m_excluded - 1] = self._D[:, :, m_excluded - 1] / self._D[:, :, -1]
                self._ST[:, :, m_excluded] = 1 - self._S[:, :, m_excluded - 1]
                DBold_m = (DBold_full[max_semi_norm[0], :, :] * DBold_full_determinant[max_semi_norm[0]]) ** (-1)
                einsum('LN, NO, lO -> Ll', self._fBold_bar_0, DBold_m, self._fBold_bar_0, out=D_m,
                             optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
                D_m -= self._f_bar_0_2
                self._S1[:, :, m_excluded] = D_m / self._D[:, :, -1]
                copyto(DBold_m_included, DBold_m_to_include)
            self._S1[:, :, 0] = self._S[:, :, 0]
            reordering[0] = (set(range(self._M))-set(reordering)).pop()   # This is the element in range(self._M) which is missing from reordering.
            return reordering
        reordering = sobol_and_reorder()

        self.reorder_data_columns(reordering)
        theta = eye(self._M, dtype=float, order=self.MEMORY_LAYOUT)
        self.write_parameters(self.Parameters(theta[reordering, :], self.Tensor3AsMatrix(self._D),
                                              self.Tensor3AsMatrix(self._S1), self.Tensor3AsMatrix(self._S), self.Tensor3AsMatrix(self._ST)))

    def _read_semi_norm(self, semi_norm_meta: Dict) -> 'Sobol.SemiNorm':
        # noinspection PyTypeChecker
        semi_norm_json = self._dir / "SemiNorm.json"
        if semi_norm_meta is None:
            if self.optimizer_options_json.exists():
                with open(semi_norm_json, mode='r') as file:
                    semi_norm_meta = json.load(file)
            else:
                semi_norm_meta = self.DEFAULT_OPTIMIZER_OPTIONS
        if semi_norm_meta['L'] != self._L:
            warn("I am changing Sobol.semi_norm.meta['L'] from {0:d} to {1:d} = Sobol.gp.L.".format(semi_norm_meta['L'], self._L))
            semi_norm_meta['L'] = self._L
        with open(semi_norm_json, mode='w') as file:
            json.dump(semi_norm_meta, file, indent=8)
        return Sobol.SemiNorm.from_meta(semi_norm_meta)

    def __init__(self, gp: GP, semi_norm: Dict = SemiNorm.DEFAULT_META):
        """ Initialize Sobol' Calculator and Optimizer.

        Args:
            gp: The underlying Gaussian process surrogate.
            semi_norm: Meta json describing a Sobol.SemiNorm.
        """

        """ Private Attributes:
        _gp (invariant): The underlying GP.
        _N, _M, _L (invariant): _gp.N, _gp.M, _gp.L.
        _lengthscale (invariant): The (M,) Array _gp.kernel.parameters.lengthscale[0, :].
        _FBold_diagonal, _2Sigma_diagonal, _2Psi_diagonal (invariant): Arrays of shape (M,) representing (M,M) diagonal matrices.

        _fBold_bar_0 (invariant): An (L,N) Matrix.
        _f_bar_0_2 (invariant): The (L,L) Matrix product of E[f] E[f.T].
        _FBold (invariant): An (M,N) Matrix.

        _objective_value: The optimization objective value (function of Theta_M_M), set by the call to Sobol.optimize()
        _objective_jacobian: The optimization objective jacobian (function of Theta_M_M), set by the call to Sobol.optimize()

        _Sigma_tilde, _Psi_tilde: (M,M) matrices.
        _FBold_tilde: An (M,N) matrix.
        """

        """ Initialize surrogate GP and related quantities, namely 
            _gp
            _N, _M, _L,: GP training data dimensions N = dataset rows (datapoints), M = input columns, L = input columns
            _lengthscale: ARD lengthscale vector
        all of which are private and invariant.
        """
        self._gp = gp
        self._N, self._M, self._L = self._gp.N, self._gp.M, self._gp.L
        self._lengthscale = self._gp.kernel.parameters.lengthscale[0, :]
        if self._lengthscale.shape != (self._M,):
            self._lengthscale = full(self._M, self._lengthscale[0], dtype=float, order=self.MEMORY_LAYOUT)

        self._Theta_M_M = zeros(self._M, dtype=float, order=self.MEMORY_LAYOUT)
        self._D = empty((self._L, self._L, self._M), dtype=float, order=self.MEMORY_LAYOUT)
        self._S1 = empty((self._L, self._L, self._M), dtype=float, order=self.MEMORY_LAYOUT)
        self._S = ones((self._L, self._L, self._M), dtype=float, order=self.MEMORY_LAYOUT)
        self._ST = zeros((self._L, self._L, self._M), dtype=float, order=self.MEMORY_LAYOUT)

        """ Declare internal calculation stages. These are documented where they are calculated, in Sobol.calculate()."""
        self._FBold_diagonal = self._2Sigma_diagonal = self._2Psi_diagonal = None
        self._FBold = self._V_pre_outer_square = self._fBold_bar_0 = self._f_bar_0_2 = None
        self._objective_value = self._objective_jacobian = None

        super().__init__(self._gp.dir / self.NAME, self.Parameters(eye(self._M, dtype=float, order=self.MEMORY_LAYOUT),
                                                                   self.Tensor3AsMatrix(self._D), self.Tensor3AsMatrix(self._S1),
                                                                   self.Tensor3AsMatrix(self._S), self.Tensor3AsMatrix(self._ST)))
        self._semi_norm = self._read_semi_norm(semi_norm)
        self.calculate()

    @classmethod
    @abstractmethod
    def from_GP(cls, fold: Fold, source_gp_name: str, destination_gp_name: str, semi_norm: Dict = SemiNorm.DEFAULT_META) -> 'Sobol':
        """ Create a Sobol object from a saved GP directory.

        Args:
            fold: The Fold housing the source and destination GPs.
            source_gp_name: The source GP directory.
            destination_gp_name: The destination GP directory. Must not exist.
            semi_norm: Meta json describing a Sobol.SemiNorm.
        Returns: The constructed Sobol object
        """
        dst = fold.dir / destination_gp_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src=fold.dir / source_gp_name, dst=dst)
        return cls(gp=GP(fold=fold, name=destination_gp_name, semi_norm=semi_norm))


# noinspection PyPep8Naming
class ROM(Model):
    """ Reduced Order Model (ROM) Calculator and optimizer.
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

    MEMORY_LAYOUT = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-All-input-is-Fortran-layout)."

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
    @abstractmethod
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
    @abstractmethod
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

    OPTIMIZED_GP_EXT = ".optimized"
    REDUCED_FOLD_EXT = ".reduced"

    """ End of required overrides."""

    @property
    def name(self) -> str:
        """ The name of this ROM."""
        return self.dir.name

    @property
    def sobol(self) -> Sobol:
        """ The Sobol object underpinning this ROM."""
        return self._sobol

    @property
    def gp(self) -> Sobol:
        """ The GP underpinning this ROM."""
        return self._gp

    @property
    def semi_norm(self) -> Sobol.SemiNorm:
        """ A Sobol.SemiNorm on the (L,L) matrix of Sobol' indices, defining the ROM optimization objective ``semi_norm(D[:,:,m])``."""
        return self._semi_norm

    def gp_name(self, iteration: int) -> str:
        """ The name of the GP produced by iteration."""
        if iteration >= 0:
            return "{0}.{1:d}".format(self.name, iteration)
        else:
            return "{0}{1}".format(self.name, self.OPTIMIZED_GB_EXT)

    def _initialize_gp(self, iteration: int) -> GP:
        if self._rbf_parameters is not None:
            gp_initializer = self.GP_Initializer.RBF
            parameters = self._rbf_parameters
            gp_rbf = self.GPType(self._fold, self.gp_name(iteration) + ".rbf", parameters)
            gp_rbf.optimize(**self._optimizer_options[-1]['gp_optimizer_options'])
            gp_dir = gp_rbf.dir.parent / self.gp_name(iteration)
            Model.copy(gp_rbf.dir, gp_dir)
            kernel = type(self._gp.kernel)(None, None, gp_dir / GP.KERNEL_NAME)
            kernel.make_ard(self._gp.M)
            return self.GPType(self._fold, self.gp_name(iteration), parameters=None)
        gp_initializer = self._optimizer_options[-1]['gp_initializer']
        parameters = self._original_parameters if gp_initializer < self.GP_Initializer.CURRENT else self._gp.parameters
        if not self._gp.kernel.is_rbf:
            if gp_initializer in (self.GP_Initializer.ORIGINAL_WITH_GUESSED_LENGTHSCALE, self.GP_Initializer.CURRENT_WITH_GUESSED_LENGTHSCALE):
                lengthscale = einsum('MK, JK -> M', self._sobol.Theta_old, self._gp.kernel.parameters.lengthscale, optimize=True, dtype=float,
                                     order=self.MEMORY_LAYOUT) * 0.5 * self._gp.M * (self._gp.M - arange(self._gp.M, dtype=float)) ** (-1)
            elif gp_initializer in (self.GP_Initializer.CURRENT_WITH_ORIGINAL_KERNEL, self.GP_Initializer.ORIGINAL):
                lengthscale = einsum('MK, JK -> M', self._Theta, self._original_parameters.kernel.parameters.lengthscale,
                                     optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            elif gp_initializer in (self.GP_Initializer.ORIGINAL_WITH_CURRENT_KERNEL, self.GP_Initializer.CURRENT):
                lengthscale = einsum('MK, JK -> M', self._sobol.Theta_old, self._gp.kernel.parameters.lengthscale, optimize=True, dtype=float,
                                     order=self.MEMORY_LAYOUT)
            parameters = parameters._replace(kernel=self._gp.kernel.Parameters(lengthscale=lengthscale))
        return self.GPType(self._fold, self.gp_name(iteration), parameters)

    def optimize(self, options: Dict):
        """ Optimize the model parameters. Do not call super().optimize, this interface only contains suggestions for implementation.

        Args:
            options: A Dict of implementation-dependent optimizer options, following the format of ROM.DEFAULT_OPTIMIZER_OPTIONS.
        """
        if options is not self._optimizer_options[-1]:
            self._optimizer_options.append(options)
            self._semi_norm = Sobol.SemiNorm.from_meta(self._optimizer_options[-1]['sobol_optimizer_options']['semi_norm'])
            self._sobol_reordering_options['semi_norm'] = self._semi_norm

        self._optimizer_options[-1]['sobol_optimizer_options']['semi_norm'] = self._semi_norm.meta
        self._write_optimizer_options(self._optimizer_options)

        iterations = self._optimizer_options[-1]['iterations']
        if iterations < 1 or self._optimizer_options[-1]['sobol_optimizer_options']['N_exploit'] < 1:
            if not iterations <= 1:
                warn("Your ROM optimization does not allow_rotation so iterations is set to 1, instead of {0:d}.".format(iterations), UserWarning)
            iterations = 1

        guess_identity_after_iteration = self._optimizer_options[-1]['guess_identity_after_iteration']
        if guess_identity_after_iteration < 0:
            guess_identity_after_iteration = iterations

        sobol_guess_identity = {**self._optimizer_options[-1]['sobol_optimizer_options'], 'N_explore': 1}
        self._Theta = self._sobol.Theta_old

        for iteration in range(iterations):
            self._gp = self._initialize_gp(iteration + 1)
            self.calculate()
            self.write_parameters(self.Parameters(
                concatenate((self.parameters.Mu, atleast_2d(self._sobol.Mu)), axis=0),
                concatenate((self.parameters.D, atleast_2d(self._semi_norm.value(self._sobol.D))), axis=0),
                concatenate((self.parameters.S1, atleast_2d(self._semi_norm.value(self._sobol.S1))), axis=0),
                concatenate((self.parameters.S, atleast_2d(self._semi_norm.value(self._sobol.S))), axis=0),
                concatenate((self.parameters.lengthscale, atleast_2d(self._sobol.lengthscale)), axis=0),
                concatenate((self.parameters.log_likelihood, atleast_2d(self._gp.log_likelihood)), axis=0)))
            if iteration < guess_identity_after_iteration:
                self._sobol.optimize(**self._optimizer_options[-1]['sobol_optimizer_options'])
            else:
                self._sobol.optimize(**sobol_guess_identity)
            self._Theta = einsum('MK, KL -> ML', self._sobol.Theta_old, self._Theta)

        self._gp = self._initialize_gp(-1)
        self.calculate()
        self._gp.test()
        self.write_parameters(self.Parameters(
            concatenate((self.parameters.Mu, atleast_2d(self._sobol.Mu)), axis=0),
            concatenate((self.parameters.D, atleast_2d(self._semi_norm.value(self._sobol.D))), axis=0),
            concatenate((self.parameters.S1, atleast_2d(self._semi_norm.value(self._sobol.S1))), axis=0),
            concatenate((self.parameters.S, atleast_2d(self._semi_norm.value(self._sobol.S))), axis=0),
            concatenate((self.parameters.lengthscale, atleast_2d(self._sobol.lengthscale)), axis=0),
            concatenate((self.parameters.log_likelihood, atleast_2d(self._gp.log_likelihood)), axis=0)))
        column_headings = ("x{:d}".format(i) for i in range(self._sobol.Mu))
        frame = Frame(self._sobol.parameters_csv.Theta, DataFrame(self._Theta, columns=column_headings))
        frame.write()

    def reduce(self, Mu: int = -1):
        """

        Args:
            Mu: The reduced dimensionality Mu &le sobol.Mu. If Mu &le 0, then Mu = sobol.Mu.

        Returns:
        """

    def calculate(self):
        """ Calculate the Model. """
        self._gp.optimize(**self._optimizer_options[-1]['gp_optimizer_options'])
        self._sobol = self.SobolType(self._gp)

    def __init__(self, name: str, sobol: Sobol, optimizer_options: Dict = DEFAULT_OPTIMIZER_OPTIONS,
                 rbf_parameters: Optional[GP.Parameters] = None):
        """ Initialize ROM object.

        Args:
            sobol: The Sobol object to construct the ROM from.
            optimizer_options: A List[Dict] similar to (and documented in) ROM.DEFAULT_OPTIMIZER_OPTIONS.
        """
        self._rbf_parameters = rbf_parameters
        self._sobol = sobol
        self._gp = sobol.gp
        self._original_parameters = self._gp.parameters._replace(kernel=self._gp.kernel.parameters)
        self._sobol_reordering_options = deepcopy(Sobol.DEFAULT_OPTIMIZER_OPTIONS)
        self._fold = Fold(self._gp.fold.dir.parent, self._gp.fold.meta['k'], self._sobol.Mu)
        self.SobolType = deepcopy(type(self._sobol))
        self.GPType = deepcopy(type(self._gp))
        if optimizer_options is None:
            super().__init__(self._fold.dir / name, None)
            self._optimizer_options = self._read_optimizer_options()
        else:
            self._optimizer_options = [optimizer_options]
            self._semi_norm = Sobol.SemiNorm.from_meta(self._optimizer_options[-1]['sobol_optimizer_options']['semi_norm'])
            self._sobol_reordering_options['semi_norm'] = self._semi_norm
            parameters = self.Parameters(Mu=self._sobol.Mu,
                                         D=self._semi_norm.value(self._sobol.D),
                                         S1=self._semi_norm.value(self._sobol.S1),
                                         S=self._semi_norm.value(self._sobol.S),
                                         lengthscale=self._sobol.lengthscale,
                                         log_likelihood=self._gp.log_likelihood)
            super().__init__(self._fold.dir / name, parameters)
            shutil.copy2(self._fold.data_csv, self.dir)
            shutil.copy2(self._fold.test_csv, self.dir)
            self.optimize(self._optimizer_options[-1])
