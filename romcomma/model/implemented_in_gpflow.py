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

""" gpflow implementation of model.base."""

from __future__ import annotations

from romcomma.typing_ import *
from romcomma.data import Fold
from romcomma.model import base
from numpy import atleast_2d, zeros, sqrt, array, transpose
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
import shutil
from enum import IntEnum, auto
from contextlib import suppress


# noinspection PyPep8Naming,PyPep8Naming
class Kernels:
    """ This is just a container for Kernel classes. Put all new Kernel classes in here."""

    class RBF(base.Kernel):
        """ Implements the RBF kernel_parameters for use with romcomma.model.implemented_in_gpflow."""

        @property
        def implemented_in(self) -> Tuple[Any, ...]:
            """ The implemented_in_??? version of this Kernel, for use in the implemented_in_??? GP.
                If ``self.variance.shape == (1,1)`` a 1-tuple of kernels is returned.
                If ``self.variance.shape == (1,L)`` an L-tuple of kernels is returned.
                If ``self.variance.shape == (L,L)`` a 1-tuple of multi-output kernels is returned.
            """
            if self.params.variance.shape[0] == 1:
                ard = (self.params.lengthscales.shape[1] == 1)
                results = tuple(gpflow.kernels.RBF(variance=self.params.variance[0, l], lengthscales=self.params.lengthscales[l])
                                for l in range(self.params.variance.shape[1]))
                # for result in results[:-1]:
                #     gpflow.set_trainable(result, False)
            else:
                raise NotImplementedError(f'Kernel.RBF is not implemented_in_gpflow for variance.shape={self.params.variance.shape}, ' +
                                          f'only for variance.shape=(1,{self.params.variance.shape[1]}) using independent gps.')
            return results

# noinspection PyPep8Naming
class GP(base.GP):
    """ Implementation of a Gaussian Process."""

    @classmethod
    @property
    def DEFAULT_OPTIONS(cls) -> Dict[str, Any]:
        """ Options passed to scipy.optimize."""
        return {'maxiter': 5000, 'gtol': 1E-16}

    def optimize(self, method: str = 'L-BFGS-B', **kwargs: Any):
        """ Optimize the GP hyper-parameters.

        Args:
            method: The optimization algorithm (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
            kwargs: A Dict of implementation-dependent optimizer options, following the format of GP.DEFAULT_OPTIMIZER_OPTIONS.
        """
        options = (self._read_options() if self._options_json.exists() else self.DEFAULT_OPTIONS)
        options.update(kwargs)
        with suppress(KeyError):
            options.pop('result')
        opt = gpflow.optimizers.Scipy()
        options.update({'result': str(tuple(opt.minimize(closure=gp.training_loss, variables=gp.trainable_variables, method=method, options=options)
                                                  for gp in self._implemented_in))})
        self._write_options(options)
        self.parameters = self._parameters.replace(noise_variance=tuple(gp.likelihood.variance.numpy() for gp in self._implemented_in),
                                                   log_marginal_likelihood=tuple(gp.log_marginal_likelihood()
                                                                                 for gp in self._implemented_in)).write()
        self._kernel.parameters = self._kernel.parameters.replace(variance=tuple(gp.kernel.variance.numpy()
                                                                                 for gp in self._implemented_in),
                                                                  lengthscales=tuple(gp.kernel.lengthscales.numpy()
                                                                                     for gp in self._implemented_in)).write()
        self._test = None

    def predict(self, X: NP.Matrix, y_instead_of_f: bool = True) -> Tuple[NP.Matrix, NP.Matrix]:
        """ Predicts the response to input X.

        Args:
            X: An (N,M) design Matrix of inputs.
            y_instead_of_f: True to include noise e in the result covariance.
        Returns: The distribution of Y or f, as a pair (mean (N, L) Matrix, std (N, L) Matrix).
        """
        results = tuple(gp.predict_y(X) if y_instead_of_f else gp.predict_f(X) for gp in self._implemented_in)
        results = tuple(transpose(result) for result in zip(*results))
        return atleast_2d(results[0]), atleast_2d(sqrt(results[1]))

    @property
    def KNoisy_Cho(self) -> TF.Tensor:
        """ The Cholesky decomposition of the LNxLN noisy kernel k(X, X)+E. """
        result = zeros(shape=(self._L * self._N, self._L * self._N))
        for l, gp in enumerate(self._implemented_in):
            X_data = gp.data[0]
            K = gp.kernel(X_data)
            K_diag = tf.linalg.diag_part(K)
            result[l*self._N:(l+1)*self._N, l*self._N:(l+1)*self._N] = tf.linalg.set_diag(K, K_diag + tf.fill(tf.shape(K_diag), gp.likelihood.variance))
        return tf.linalg.cholesky(result)

    @property
    def KNoisyInv_Y(self) -> TF.Tensor:
        """ The LN-Vector, which pre-multiplied by the LoxLN kernel k(x, X) gives the Lo-Vector predictive mean fBar(x).
        Returns: ChoSolve(self.KNoisy_Cho, self.Y) """
        Y_data = self._Y.transpose().flatten()
        return tf.linalg.cholesky_solve(tf.linalg.cholesky(self.KNoisy_Cho), Y_data)

    def _check_KNoisyInv_Y(self, x: NP.Matrix) -> TF.Tensor:
        """ FOR TESTING PURPOSES ONLY. Should return 0 Vector (to within numerical error tolerance).

        Args:
            x: An (o, M) matrix of inputs.
        Returns: Should return zeros((Lo)) (to within numerical error tolerance)

        """
        o = x.shape[0]
        kernel = zeros(shape=(self._L * o, self._L * self._N))
        for l, gp in enumerate(self._implemented_in):
            X_data = gp.data[0]
            kernel[l*o:(l+1)*o, l*self._N:(l+1)*self._N] = gp.kernel(X_data, x)
        predicted = self.predict(x)[0]
        return predicted - tf.einsum('on, n -> o', kernel, self.KNoisyInv_Y)

    def __init__(self, name: str, fold: Fold, is_read: bool, is_isotropic: bool, is_independent: bool,
                 kernel_parameters: Optional[base.Kernel.Parameters] = None, **kwargs: NP.Matrix):
        """ GP Constructor. Calls model.__init__ to setup parameters, then checks dimensions.

        Args:
            name: The name of this GP.
            fold: The Fold housing this GP.
            is_read: If True, the GP.kernel.parameters and GP.parameters and are read from ``fold.folder/name``, otherwise defaults are used.
            is_independent: Whether the outputs will be treated as independent.
            is_isotropic: Whether to restrict the kernel to be isotropic.
            kernel_parameters: A base.Kernel.Parameters to use for GP.kernel.parameters. If not None, this replaces the kernel specified by file/defaults.
                If None, the kernel is read from file, or set to the default base.Kernel.Parameters(), according to read_from_file.
            **kwargs: The GP.parameters fields=values to replace after reading from file/defaults.
        Raises:
            IndexError: If a parameter is mis-shaped.
        """
        super().__init__(name, fold, is_read, is_isotropic, is_independent, kernel_parameters, **kwargs)
        # e = gpflow.Parameter(value=self._parameters.values.e,
        #                      transform=tfp.bijectors.Chain([tfp.bijectors.Shift(self._parameters.values.e_floor), tfp.bijectors.Softplus()]), trainable=True)
        self._implemented_in = tuple(gpflow.models.GPR(data=(self._X, self._Y[:, [l]]), kernel=kernel, mean_function=None,
                                                       noise_variance=self.params.noise_variance[0, l]) for l, kernel in enumerate(self._kernel.implemented_in))


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
        def from_meta(cls, meta: Union[Dict, 'Sobol.SemiNorm']) -> Sobol.SemiNorm:
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
                raise TypeError("SemiNorm meta data must be a Dict or a SemiNorm, not a {0}.".format(type(meta)))
            if meta['classmethod'] == 'element':
                return cls.element(meta['L'], **meta['kwargs'])
            else:
                raise NotImplementedError("Unrecognized meta['classmethod'] = '{0}'. ".format(meta['classmethod']) +
                                          "Please implement the relevant @classmethod in Sobol.SemiNorm " +
                                          "and register it in Sobol.SemiNorm.from_meta().")

        @classmethod
        def element(cls, L: int, row: int, column: int) -> Sobol.SemiNorm:
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
    def from_GP(cls, fold: Fold, source_gp_name: str, destination_gp_name: str, Mu: int = -1, read_parameters: bool = False) -> Sobol:
        """ Create a Sobol object from a saved GP folder.

        Args:
            fold: The Fold housing the source and destination gps.
            source_gp_name: The source GP folder.
            destination_gp_name: The destination GP folder. Must not exist.
            Mu: The dimensionality of the rotated input basis u. If this is not in range(1, fold.M+1), Mu=fold.M is used.
            read_parameters: True to store read the existing parameters and store them in self.parameters_read (for information purposes only).

        Returns: The constructed Sobol object
        """
        dst = fold.folder / destination_gp_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src=fold.folder / source_gp_name, dst=dst)
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
                                           ('lengthscales', NP.Matrix), ('log_marginal_likelihood', NP.Matrix)])
    """ 
        **Mu** -- A numpy [[int]] specifying the number of input dimensions in the rotated basis u.

        **D** -- An (L L, M) Matrix of cumulative conditional variances D[l,k,m] = S[l,k,m] D[l,k,M].

        **S1** -- An (L L, M) Matrix of Sobol' main indices.

        **S** -- An (L L, M) Matrix of Sobol' cumulative indices.

        **lengthscales** -- A (1,M) Covector of RBF lengthscales, or a (1,1) RBF lengthscales.

        **log_marginal_likelihood** -- A numpy [[float]] used to record the log marginal likelihood.
    """
    DEFAULT_PARAMETERS = Parameters(*(atleast_2d(None),) * 6)

    DEFAULT_OPTIMIZER_OPTIONS = {'iterations': 1, 'guess_identity_after_iteration': 1, 'sobol_options': Sobol.DEFAULT_OPTIMIZER_OPTIONS,
                                 'gp_initializer': GP_Initializer.CURRENT_WITH_GUESSED_LENGTHSCALE,
                                 'gp_options': GP.DEFAULT_OPTIONS}
    """ 
        **iterations** -- The number of ROM iterations. Each ROM iteration essentially calls Sobol.optimimize(options['sobol_options']) 
            followed by GP.optimize(options['gp_options'])).

        **sobol_options*** -- A Dict of Sobol optimizer options, similar to (and documented in) Sobol.DEFAULT_OPTIMIZER_OPTIONS.

        **guess_identity_after_iteration** -- After this many ROM iterations, Sobol.optimize does no exploration, 
            just gradient descending from Theta = Identity Matrix.

        **reuse_original_gp** -- True if GP.optimize is initialized each time from the GP originally provided.

        **gp_options** -- A Dict of GP optimizer options, similar to (and documented in) GP.DEFAULT_OPTIMIZER_OPTIONS.
    """

    @classmethod
    def from_ROM(cls, fold: Fold, name: str, suffix: str = ".0", Mu: int = -1, rbf_parameters: Optional[GP.Parameters] = None) -> ROM:
        """ Create a ROM object from a saved ROM folder.

        Args:
            fold: The Fold housing the ROM to load.
            name: The name of the saved ROM to create from.
            suffix: The suffix to append to the most optimized gp.
            Mu: The dimensionality of the rotated input basis u. If this is not in range(1, fold.M+1), Mu=fold.M is used.

        Returns: The constructed ROM object
        """
        optimization_count = [optimized.name.count(cls.OPTIMIZED_GB_EXT) for optimized in fold.folder.glob(name + cls.OPTIMIZED_GB_EXT + "*")]
        source_gp_name = name + cls.OPTIMIZED_GB_EXT * max(optimization_count)
        destination_gp_name = source_gp_name + suffix
        return cls(name=name,
                   sobol=Sobol.from_GP(fold, source_gp_name, destination_gp_name, Mu=Mu, read_parameters=True),
                   options=None, rbf_parameters=rbf_parameters)

    @classmethod
    def from_GP(cls, fold: Fold, name: str, source_gp_name: str, options: Dict, Mu: int = -1,
                rbf_parameters: Optional[GP.Parameters] = None) -> ROM:
        """ Create a ROM object from a saved GP folder.

        Args:
            fold: The Fold housing the ROM to load.
            name: The name of the saved ROM to create from.
            source_gp_name: The source GP folder.
            Mu: The dimensionality of the rotated input basis u. If this is not in range(1, fold.M+1), Mu=fold.M is used.
            options: A Dict of ROM optimizer options.

        Returns: The constructed ROM object
        """
        return cls(name=name,
                   sobol=Sobol.from_GP(fold=fold, source_gp_name=source_gp_name, destination_gp_name=name + ".0", Mu=Mu),
                   options=options, rbf_parameters=rbf_parameters)

    OPTIMIZED_GB_EXT = ".optimized"
    REDUCED_FOLD_EXT = ".reduced"

    """ End of required overrides."""

