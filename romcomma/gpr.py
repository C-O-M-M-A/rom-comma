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

""" Contains:

A GPInterface base class - Anyone wishing to implement their own GPs should inherit from this).

A GPFlow implementation of Gaussian Process Regression.
"""

from __future__ import annotations

from abc import abstractmethod
from romcomma.typing_ import *
from romcomma.data import Fold, Frame
from romcomma.base import Parameters, Model
from romcomma. kernels import Kernel
from numpy import atleast_2d, zeros, sqrt, transpose
import gpflow as gf
import tensorflow as tf
from contextlib import suppress


# noinspection PyPep8Naming
class GPInterface(Model):
    """ Interface to a Gaussian Process."""

    class Parameters(Parameters):
        """ Abstraction of the parameters of a GP."""

        @classmethod
        @property
        def Values(cls) -> Type[NamedTuple]:
            """ The NamedTuple underpinning this Parameters set."""
            class Values(NamedTuple):
                """ Abstraction of the parameters of a GP.

                Attributes:
                    noise_variance (NP.Matrix): An (L,L), (1,L) or (1,1) noise variance matrix. (1,L) represents an (L,L) diagonal matrix.
                    kernel (NP.Matrix): A numpy [[str]] identifying the type of Kernel, as returned by gp.kernel.TypeIdentifier(). This is never set externally.
                        The kernel parameter, when provided, must be a [[Kernel.Parameters]] storing the desired kernel parameters.
                        The kernel is constructed and its type inferred from these parameters.
                    log_marginal_likelihood (NP.Matrix): A numpy [[float]] used to record the log marginal likelihood. This is an output parameter, not input.
                """
                noise_variance: NP.Matrix = atleast_2d(0.9)
                kernel: NP.Matrix = atleast_2d(None)
                log_marginal_likelihood: NP.Matrix = atleast_2d(1.0)
            return Values

    @classmethod
    @property
    @abstractmethod
    def DEFAULT_OPTIONS(cls) -> Dict[str, Any]:
        """ Default hyper-parameter optimizer options"""

    @classmethod
    @property
    def KERNEL_DIR_NAME(cls) -> str:
        """ The name of the folder where kernel parameters are stored."""
        return "kernel"

    @property
    def fold(self) -> Fold:
        """ The parent fold """
        return self._fold

    @property
    def test_csv(self) -> Path:
        return self._folder / "__test__.csv"

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
    def kernel(self) -> Kernel:
        """ The GP Kernel. """
        return self._kernel

    @property
    @abstractmethod
    def implementation(self) -> Tuple[Any, ...]:
        """ The implementation of this GP in GPFlow.
            If ``self.variance.shape == (1,L)`` an L-tuple of kernels is returned.
            If ``self.variance.shape == (L,L)`` a 1-tuple of multi-output kernels is returned.
        """

    @property
    @abstractmethod
    def KNoisy_Cho(self) -> Union[NP.Matrix, TF.Tensor]:
        """ The Cholesky decomposition of the LNxLN noisy kernel k(X, X)+E. """

    @property
    @abstractmethod
    def KNoisyInv_Y(self) -> Union[NP.Matrix, TF.Tensor]:
        """ The split_axis_shape-Vector, which pre-multiplied by the LoxLN kernel k(x, X) gives the Lo-Vector predictive mean fBar(x).
        Returns: ChoSolve(self.KNoisy_Cho, self.Y) """

    @abstractmethod
    def optimize(self, method: str = 'L-BFGS-B', **kwargs: Any):
        """ Optimize the GP hyper-parameters.

        Args:
            method: The optimization algorithm (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
            kwargs: A Dict of implementation-dependent optimizer options, following the format of GP.DEFAULT_OPTIMIZER_OPTIONS.
        """

    @abstractmethod
    def predict(self, X: NP.Matrix, y_instead_of_f: bool = True) -> Tuple[NP.Matrix, NP.Matrix]:
        """ Predicts the response to input X.

        Args:
            X: A (o, M) design Matrix of inputs.
            y_instead_of_f: True to include noise e in the result covariance.
        Returns: The distribution of Y or f, as a pair (mean (o, L) Matrix, std (o, L) Matrix).
        """

    def test(self) -> Frame:
        """ Tests the GP on the test_data data in GP.fold.test_csv.

        Returns: The test_data results as a Frame backed by GP.test_result_csv.
        """
        if self._test is None:
            self._test = Frame(self.test_csv, self._fold.test_data.df)
            Y_heading = self._fold.meta['data']['Y_heading']
            result = self.predict(self._fold.test_x.values)
            predictive_mean = (self._test.df.loc[:, [Y_heading]].copy().rename(columns={Y_heading: "Predictive Mean"}, level=0))
            predictive_mean.iloc[:] = result[0]
            predictive_std = (self._test.df.loc[:, [Y_heading]].copy().rename(columns={Y_heading: "Predictive Std"}, level=0))
            predictive_std.iloc[:] = result[1]
            self._test.df = self._test.df.join([predictive_mean, predictive_std])
            self._test.write()
        return self._test

    def calculate(self):
        """ Fit the GP to the training data, which is actually automatic. """
        self._test = None

    def broadcast_parameters(self, is_independent: bool, is_isotropic: bool, folder: Optional[PathLike] = None) -> GP:
        """ Broadcast the parameters of the GP (including kernels) to highr dimensions. Shrinkage raises errors, unchanged dimensions silently nop.

        Args:
            is_independent: Whether the outputs will be treated as independent.
            is_isotropic: Whether to restrict the kernel to be isotropic.
            folder: The file location, which is ``self.folder`` if ``folder is None`` (the default).
        Returns: ``self``, for chaining calls.
        """
        target_shape = (1, self._L) if is_independent else (self._L, self._L)
        self._parameters.broadcast_value(model_name=self.folder, field="noise_variance", target_shape=target_shape, is_diagonal=is_independent, folder=folder)
        self._kernel.broadcast_parameters(variance_shape=target_shape, M=1 if is_isotropic else self._M, folder=folder)

    @abstractmethod
    def __init__(self, name: str, fold: Fold, is_read: bool, is_isotropic: bool, is_independent: bool,
                 kernel_parameters: Optional[Kernel.Parameters] = None, **kwargs: NP.Matrix):
        """ GP Constructor. Calls __init__ to setup parameters, then checks dimensions.

        Args:
            name: The name of this GP.
            fold: The Fold housing this GP.
            is_read: If True, the GP.kernel.parameters and GP.parameters and are read from ``fold.folder/name``, otherwise defaults are used.
            is_independent: Whether the outputs will be treated as independent.
            is_isotropic: Whether to restrict the kernel to be isotropic.
            kernel_parameters: A Kernel.Parameters to use for GP.kernel.parameters. If not None, this replaces the kernel specified by file/defaults.
                If None, the kernel is read from file, or set to the default Kernel.Parameters(), according to read_from_file.
            **kwargs: The GP.parameters fields=values to replace after reading from file/defaults.
        Raises:
            IndexError: If a parameter is mis-shaped.
        """
        self._fold, self._folder = fold, fold.folder / name
        self._X, self._Y = self._fold.X.to_numpy(dtype=float, copy=True), self._fold.Y.to_numpy(dtype=float, copy=True)
        self._N, self._M, self._L = self._fold.N, self._fold.M, self._fold.L
        super().__init__(self._folder, is_read, **kwargs)
        if is_read and kernel_parameters is None:
            KernelType = Kernel.TypeFromIdentifier(self.params.values.kernel[0, 0])
            self._kernel = KernelType(self._folder / self.KERNEL_DIR_NAME, is_read)
        else:
            if kernel_parameters is None:
                kernel_parameters = Kernel.Parameters()
            KernelType = Kernel.TypeFromParameters(kernel_parameters)
            self._kernel = KernelType(self._folder / self.KERNEL_DIR_NAME, is_read, **kernel_parameters.as_dict())
            self._parameters.replace(kernel=atleast_2d(KernelType.TYPE_IDENTIFIER)).write()
        self.broadcast_parameters(is_independent, is_isotropic)
        self._implementation = self.implementation


# noinspection PyPep8Naming
class GP(GPInterface):
    """ Implementation of a Gaussian Process."""

    @classmethod
    @property
    def DEFAULT_OPTIONS(cls) -> Dict[str, Any]:
        return {'maxiter': 5000, 'gtol': 1E-16}

    @property
    def implementation(self) -> Tuple[Any, ...]:
        return tuple(gf.models.GPR(data=(self._X, self._Y[:, [l]]), kernel=kernel, mean_function=None, noise_variance=self.params.noise_variance[0, l])
                     for l, kernel in enumerate(self._kernel.implementation))

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
        opt = gf.optimizers.Scipy()
        options.update({'result': str(tuple(opt.minimize(closure=gp.training_loss, variables=gp.trainable_variables, method=method, options=options)
                                                  for gp in self._implementation))})
        self._write_options(options)
        self.parameters = self._parameters.replace(noise_variance=tuple(gp.likelihood.variance.numpy() for gp in self._implementation),
                                                   log_marginal_likelihood=tuple(gp.log_marginal_likelihood()
                                                                                 for gp in self._implementation)).write()
        self._kernel.parameters = self._kernel.parameters.replace(variance=tuple(gp.kernel.variance.numpy()
                                                                                 for gp in self._implementation),
                                                                  lengthscales=tuple(gp.kernel.lengthscales.numpy()
                                                                                     for gp in self._implementation)).write()
        self._test = None

    def predict(self, X: NP.Matrix, y_instead_of_f: bool = True) -> Tuple[NP.Matrix, NP.Matrix]:
        results = tuple(gp.predict_y(X) if y_instead_of_f else gp.predict_f(X) for gp in self._implementation)
        results = tuple(transpose(result) for result in zip(*results))
        return atleast_2d(results[0]), atleast_2d(sqrt(results[1]))

    @property
    def KNoisy_Cho(self) -> TF.Tensor:
        result = zeros(shape=(self._L * self._N, self._L * self._N))
        for l, gp in enumerate(self._implementation):
            X_data = gp.data[0]
            K = gp.kernel(X_data)
            K_diag = tf.linalg.diag_part(K)
            result[l*self._N:(l+1)*self._N, l*self._N:(l+1)*self._N] = tf.linalg.set_diag(K, K_diag + tf.fill(tf.shape(K_diag), gp.likelihood.variance))
        return tf.linalg.cholesky(result)

    @property
    def KNoisyInv_Y(self) -> TF.Tensor:
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
        for l, gp in enumerate(self._implementation):
            X_data = gp.data[0]
            kernel[l*o:(l+1)*o, l*self._N:(l+1)*self._N] = gp.kernel(X_data, x)
        predicted = self.predict(x)[0]
        return predicted - tf.einsum('on, n -> o', kernel, self.KNoisyInv_Y)

    def __init__(self, name: str, fold: Fold, is_read: bool, is_isotropic: bool, is_independent: bool,
                 kernel_parameters: Optional[Kernel.Parameters] = None, **kwargs: NP.Matrix):
        """ GP Constructor. Calls __init__ to setup parameters, then checks dimensions.

        Args:
            name: The name of this GP.
            fold: The Fold housing this GP.
            is_read: If True, the GP.kernel.parameters and GP.parameters and are read from ``fold.folder/name``, otherwise defaults are used.
            is_independent: Whether the outputs will be treated as independent.
            is_isotropic: Whether to restrict the kernel to be isotropic.
            kernel_parameters: A Kernel.Parameters to use for GP.kernel.parameters. If not None, this replaces the kernel specified by file/defaults.
                If None, the kernel is read from file, or set to the default Kernel.Parameters(), according to read_from_file.
            **kwargs: The GP.parameters fields=values to replace after reading from file/defaults.
        Raises:
            IndexError: If a parameter is mis-shaped.
        """
        super().__init__(name, fold, is_read, is_isotropic, is_independent, kernel_parameters, **kwargs)
