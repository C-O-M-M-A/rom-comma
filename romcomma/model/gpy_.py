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

""" GPy implementation of model.base."""

from romcomma.typing_ import Optional, NP, PathLike, NamedTuple, Tuple, Dict, Union, Callable
from romcomma.data import Fold
from romcomma.model import base
from numpy import atleast_2d, atleast_3d, transpose, zeros, einsum, sqrt, array, full
import GPy
import shutil
from enum import IntEnum, auto


# noinspection PyPep8Naming,PyPep8Naming
class Kernel:
    """ This is just a container for Kernel classes. Put all new Kernel classes in here."""

    class ExponentialQuadratic(base.Kernel):
        """ Implements the exponential quadratic kernel for use with romcomma.gpy_."""

        """ Required overrides."""

        MEMORY_LAYOUT = 'C'
        Parameters = NamedTuple("Parameters", [("lengthscale", NP.Matrix)])
        """
            **lengthscale** -- A (1,M) Covector of ARD lengthscales, or a (1,1) RBF lengthscale.
        """
        DEFAULT_PARAMETERS = Parameters(lengthscale=atleast_2d(0.2))

        """ End of required overrides."""

        @property
        def is_rbf(self) -> bool:
            """ Returns True if kernel is RBF, False if it is ARD. """
            return self._is_rbf

        def make_ard(self, M):
            if self._is_rbf:
                self.write_parameters(self.parameters._replace(lengthscale=full((1, M), self.parameters.lengthscale)))
                self._is_rbf = False

        def gpy(self, f: float) -> GPy.kern.RBF:
            """ Returns the GPy version of this kernel."""
            return GPy.kern.RBF(self._M, f, self._parameters.lengthscale[0, 0], False) if self._is_rbf \
                else GPy.kern.RBF(self._M, f, self._parameters.lengthscale[0, :self._M], True)

        def calculate(self):
            """ This function is an interface requirement which does nothing."""

        @property
        def matrix(self) -> NP.Matrix:
            """ NOT FOR PUBLIC USE."""
            return super().matrix

        def __init__(self, X0: Optional[NP.Matrix], X1: Optional[NP.Matrix], dir_: PathLike = "", parameters: Optional[Parameters] = None):
            """ Construct a Kernel.

            Args:
                X0: An (N0,M) Design (feature) Matrix. Use None if and only if kernel is only for recording parameters.
                X1: An (N1,M) Design (feature) Matrix. Use None if and only if kernel is only for recording parameters.
                dir_: The kernel file location. If and only if this is empty, kernel.with_frames=False
                parameters: The kernel parameters. If None these are read from dir_.
            """
            super().__init__(X0, X1, dir_, parameters)
            self._is_rbf = (self.parameters.lengthscale.shape[1] == 1)
            assert self.with_frames or (self._is_rbf or self._parameters.lengthscale.shape[1] >= self._M), \
                "This ARD kernel has {0:d} lengthscale parameters when M={1:d}.".format(self._parameters.lengthscale.shape[1], self._M)


# noinspection PyPep8Naming
class GP(base.GP):
    """ Implementation of a Gaussian Process."""

    """ Required overrides."""

    MEMORY_LAYOUT = 'C'

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
        return -abs(self._gpy.log_likelihood())

    def calculate(self):
        """ Fit the GP to the training data. """
        self._test = None

    def predict(self, X: NP.Matrix, Y_instead_of_F: bool = True) -> Tuple[NP.Matrix, NP.Matrix, NP.Tensor3]:
        """ Predicts the response to input X.

        Args:
            X: An (N,M) design Matrix of inputs.
            Y_instead_of_F: True to include noise e in the result covariance.
        Returns: The distribution of Y or f, as a triplet (mean (N, L) Matrix, std (N, L) Matrix), covariance (N, L, L) Tensor3.
        """
        result = self._gpy.predict(X, include_likelihood=Y_instead_of_F)
        return result[0], sqrt(result[1]), transpose(atleast_3d(result[1]), [1, 2, 0])

    def optimize(self, **kwargs):
        """ Optimize the GP hyper-parameters.

        Args:
            options: A Dict of implementation-dependent optimizer options, following the format of GP.DEFAULT_OPTIMIZER_OPTIONS.
        """
        if kwargs is None:
            kwargs = self._read_optimizer_options() if self.optimizer_options_json.exists() else self.DEFAULT_OPTIMIZER_OPTIONS
        self._gpy.optimize(**kwargs)
        if self._gpy.Gaussian_noise.variance[0][0] < self.parameters.e_floor[0, 0]:
            self._gpy.Gaussian_noise.variance = self.parameters.e_floor[0, 0]
            self._gpy.Gaussian_noise.variance.fix()
            self._gpy.optimize(**kwargs)
        """ Update and record GP and kernel parameters, using Model.write_parameters(new_parameters)."""
        self._write_optimizer_options(kwargs)
        self.write_parameters(self._parameters._replace(f=array(self._gpy.rbf.variance), e=array(self._gpy.Gaussian_noise.variance),
                                                        log_likelihood=self.log_likelihood))
        self._kernel.write_parameters(self._kernel.parameters._replace(lengthscale=array(self._gpy.rbf.lengthscale)))
        self._test = None

    @property
    def Kinv_Y(self) -> NP.Matrix:
        """ The (N,L) Matrix (K(X,X) + e I)^(-1) Y."""
        return atleast_2d(self._gpy.posterior.woodbury_vector).reshape((self._N, self._L))

    def _check_Kinv_Y(self, x: NP.Matrix, Y_instead_of_F: bool = True) -> NP.Vector:
        """ FOR TESTING PURPOSES ONLY. Should return 0 Vector (to within numerical error tolerance)."""
        kern = self._gpy.kern.K(x, self.X)
        __Kinv_Y = self.Kinv_Y
        result = self._gpy.predict(x, include_likelihood=Y_instead_of_F)[0]
        result -= einsum('in, nj -> ij', kern, __Kinv_Y)
        return result

    def _validate_parameters(self):
        """ Generic and specific validation.

        Raises:
            IndexError: (generic) if parameters.kernel and parameters.e_floor are not shaped (1,1).
            IndexError: (generic) unless parameters.f.shape == parameters.e == (1,1) or (L,L).
            IndexError: (specific) unless parameters.f.shape == parameters.e == (1,1).
        """
        super()._validate_parameters()
        if self.parameters.f.shape[0] != 1:
            raise IndexError("GPy will only accept (1,1) parameters.f and parameters.e, not ({0:d},{0:d})".format(self.parameters.f.shape[0]))

    def __init__(self, fold: Fold, name: str, parameters: Optional[base.GP.Parameters] = None):
        """ GP Constructor. Calls Model.__Init__ to setup parameters, then checks dimensions.

        Args:
            fold: The Fold housing this GaussianProcess.
            name: The name of this GaussianProcess.
            parameters: The model parameters. If None these are read from fold/name, otherwise they are written to fold/name.
        """
        super().__init__(fold, name, parameters)
        self._gpy = GPy.models.GPRegression(self.X, self.Y, self._kernel.gpy(self.parameters.f), noise_var=self.parameters.e)


# noinspection PyPep8Naming
class Sobol(base.Sobol):
    """ Interface to a Sobol' Index Calculator and Optimizer.

    Internal quantities are called variant if they depend on Theta, invariant otherwise.
    Invariants are calculated in the constructor. Variants are calculated in Theta.setter."""

    """ Required overrides."""

    MEMORY_LAYOUT = 'C'

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

    MEMORY_LAYOUT = 'C'

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
        optimization_count = [optimized.name.count(cls.OPTIMIZED_GB_EXT) for optimized in fold.dir.glob(name + cls.OPTIMIZED_GB_EXT + "*")]
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

