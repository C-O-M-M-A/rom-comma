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

""" Contains the MOGP class implementing Gaussian Process Regression. """

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.data.storage import Fold, Frame
from romcomma.base.classes import Parameters, Model
from romcomma.gpr.kernels import Kernel


class Likelihood(Model):

    class Parameters(Parameters):
        """ The Parameters set of a MOGP."""

        @classmethod
        @property
        def Values(cls) -> Type[NamedTuple]:
            """ The NamedTuple underpinning this Parameters set."""

            class Values(NamedTuple):
                """ The parameters set of a MOGP.

                Attributes:
                    variance (NP.Matrix): An (L,L), (1,L) or (1,1) noise variance matrix. (1,L) represents an (L,L) diagonal matrix.
                    log_marginal (NP.Matrix): A numpy [[float]] used to record the log marginal likelihood. This is an output parameter, not input.
                """
                variance: NP.Matrix = np.atleast_2d(0.0001)
                log_marginal: NP.Matrix = np.atleast_2d(1.0)

            return Values

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        return {'variance': True, 'covariance': True}

    @classmethod
    @property
    def VARIANCE_FLOOR(cls) -> Dict[str, Any]:
        return 1.0001E-6

    @property
    def is_covariant(self) -> bool:
        return self.params.variance.shape[0] > 1

    def optimize(self, **kwargs) -> Dict[str, Any]:
        """ Merely sets the trainable parameters."""
        options = self.OPTIONS | kwargs
        if self.is_covariant:
            gf.set_trainable(self._parent._implementation[0].likelihood.variance._cholesky_diagonal, options['variance'])
            gf.set_trainable(self._parent._implementation[0].likelihood.variance._cholesky_lower_triangle, options['covariance'])
        else:
            for implementation in self._parent.implementation:
                gf.set_trainable(implementation.likelihood.variance, options['variance'])
        return options

    def __init__(self, parent: GPR, read_parameters: bool = False, **kwargs: NP.Matrix):
        super().__init__(parent.folder / 'likelihood', read_parameters, **kwargs)
        self._parent = parent


# noinspection PyPep8Naming
class GPR(Model):
    """ Interface to a Gaussian Process."""

    class Parameters(Parameters):
        """ The Parameters set of a MOGP."""

        @classmethod
        @property
        def Values(cls) -> Type[NamedTuple]:
            """ The NamedTuple underpinning this Parameters set."""
            class Values(NamedTuple):
                """ The parameters set of a MOGP.

                Attributes:
                    kernel (NP.Matrix): A numpy [[str]] identifying the type of Kernel, as returned by gp.kernel.TypeIdentifier(). This is never set externally.
                        The kernel parameter, when provided, must be a ``[[Kernel.Parameters]]`` set storing the desired kernel parameters.
                        The kernel is constructed by inferring its type from the type of Kernel.Parameters.
                """
                kernel: NP.Matrix = np.atleast_2d(None)
            return Values

    @classmethod
    @property
    @abstractmethod
    def OPTIONS(cls) -> Dict[str, Any]:
        """ Hyper-parameter optimizer options"""

    @classmethod
    @property
    def KERNEL_FOLDER_NAME(cls) -> str:
        """ The name of the folder where kernel parameters are stored."""
        return "kernel"

    @property
    def fold(self) -> Fold:
        """ The parent fold. """
        return self._fold

    @property
    def test_csv(self) -> Path:
        return self._folder / 'test.csv'

    @property
    def test_summary_csv(self) -> Path:
        return self._folder / "test_summary.csv"

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    @property
    def likelihood(self) -> Likelihood:
        return self._likelihood

    @property
    @abstractmethod
    def implementation(self) -> Tuple[Any, ...]:
        """ The implementation of this MOGP in GPFlow.
            If ``noise_variance.shape == (1,L)`` an L-tuple of kernels is returned.
            If ``noise_variance.shape == (L,L)`` a 1-tuple of multi-output kernels is returned.
        """

    @property
    def L(self) -> int:
        """ The output (Y) dimensionality."""
        return self._L

    @property
    def M(self) -> int:
        """ The input (X) dimensionality."""
        return self._M

    @property
    def N(self) -> int:
        """ The the number of training samples."""
        return self._N


    @property
    @abstractmethod
    def X(self) -> Any:
        """ The implementation training inputs."""

    @property
    @abstractmethod
    def Y(self) -> Any:
        """ The implementation training outputs."""

    @property
    @abstractmethod
    def K_cho(self) -> Union[NP.Matrix, TF.Tensor]:
        """ The Cholesky decomposition of the LNxLN noisy kernel(X, X) + likelihood.variance. Shape is (L,N,N) if self.kernel.is_covariant, else (LN, LN)."""

    @property
    @abstractmethod
    def K_inv_Y(self) -> Union[NP.Matrix, TF.Tensor]:
        """ The LN-Vector, which pre-multiplied by the LoxLN kernel k(x, X) gives the Lo-Vector predictive mean f(x).
        Shape is (L,1,N) self.kernel.is_covariant, else (1, L, N).
        Returns: ChoSolve(self.K_cho, self.Y) """

    @abstractmethod
    def optimize(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: NP.Matrix, y_instead_of_f: bool = True) -> Tuple[NP.Matrix, NP.Matrix]:
        """ Predicts the response to input X.

        Args:
            X: An (o, M) design Matrix of inputs.
            y_instead_of_f: True to include noise in the variance of the result.
        Returns: The distribution of Y or f, as a pair (mean (o, L) Matrix, std (o, L) Matrix).
        """

    def test(self) -> Frame:
        """ Tests the MOGP on the test data in self._fold.test_data. Test results comprise three values for each output at each sample:
        The mean prediction, the std error of prediction and the Z score of prediction (i.e. error of prediction scaled by std error of prediction).

        Returns: The test_data results as a Frame backed by MOGP.test_result_csv.
        """
        result = Frame(self.test_csv, self._fold.test_data.df)
        Y_heading = self._fold.meta['data']['Y_heading']
        prediction = self.predict(self._fold.test_x.values)
        predictive_mean = result.df.loc[:, [Y_heading]].copy().rename(columns={Y_heading: 'Predictive Mean'}, level=0)
        predictive_mean.iloc[:] = prediction[0]
        predictive_std = result.df.loc[:, [Y_heading]].copy().rename(columns={Y_heading: 'Predictive Std'}, level=0)
        predictive_std.iloc[:] = prediction[1]
        predictive_score = result.df.loc[:, [Y_heading]].copy().rename(columns={Y_heading: 'Predictive Z Score'}, level=0)
        predictive_score.iloc[:] -= predictive_mean.to_numpy(dtype=float, copy=False)
        rmse = predictive_score.iloc[:].copy().rename(columns={'Predictive Z Score': 'RMSE'}, level=0)
        predictive_score.iloc[:] /= predictive_std.to_numpy(dtype=float, copy=False)
        result.df = result.df.join([predictive_mean, predictive_std, predictive_score])
        result.write()
        rmse = rmse**2
        rmse = rmse.sum(axis=0)/rmse.count(axis=0)
        r2 = 1 - rmse
        rmse = rmse**(1/2)
        rmse = rmse if isinstance(rmse, pd.DataFrame) else pd.DataFrame(rmse).transpose()
        r2 = r2 if isinstance(r2, pd.DataFrame) else pd.DataFrame(r2).transpose()
        r2 = r2.rename(columns={'RMSE': 'R^2'}, level=0)
        predictive_std = predictive_std.sum(axis=0)/predictive_std.count(axis=0)
        predictive_std = predictive_std if isinstance(predictive_std, pd.DataFrame) else pd.DataFrame(predictive_std).transpose()
        ci = (predictive_std.iloc[:].copy().rename(columns={'Predictive Std': '95% CI'}, level=0))
        ci = ci * 2
        outliers = predictive_score[predictive_score**2 > 4].count(axis=0)/predictive_score.count(axis=0)
        outliers = outliers if isinstance(outliers, pd.DataFrame) else pd.DataFrame(outliers).transpose()
        outliers = outliers.rename(columns={'Predictive Z Score': 'outliers'})
        summary = rmse.join([r2, predictive_std, ci, outliers])
        summary = Frame(self.test_summary_csv, summary)
        return result

    def broadcast_parameters(self, is_covariant: bool, is_isotropic: bool, folder: Optional[PathLike] = None) -> GPR:
        """ Broadcast the parameters of the MOGP (including kernels) to higher dimensions.
        Shrinkage raises errors, unchanged dimensions silently do nothing.

        Args:
            is_covariant: Whether the outputs will be treated as dependent.
            is_isotropic: Whether to restrict the kernel to be isotropic.
            folder: The file location, which is ``self.folder`` if ``folder is None`` (the default).
        Returns: ``self``, for chaining calls.
        """
        target_shape = (self._L, self._L) if is_covariant else (1, self._L)
        self._likelihood.parameters.broadcast_value(model_name=self.folder, field="variance", target_shape=target_shape, is_diagonal=True,
                                                    folder=folder)
        self._kernel.broadcast_parameters(variance_shape=target_shape, M=1 if is_isotropic else self._M, folder=folder)
        self._implementation = None
        self._implementation = self.implementation
        return self

    def __init__(self, name: str, fold: Fold, is_read: bool | None, is_covariant: bool, is_isotropic: bool,
                 kernel_parameters: Kernel.Parameters | None = None, likelihood_variance: NP.Matrix | None = None):
        """ Set up parameters, and checks dimensions.

        Args:
            name: The name of this MOGP.
            fold: The Fold housing this MOGP.
            is_read: If True, the MOGP.kernel.parameters and MOGP.parameters and are read from ``fold.folder/name``, otherwise defaults are used.
            is_covariant: Whether the outputs will be treated as independent.
            is_isotropic: Whether to restrict the kernel to be isotropic.
            kernel_parameters: A Kernel.Parameters to use for MOGP.kernel.parameters. If not None, this replaces the kernel specified by file/defaults.
                If None, the kernel is read from file, or set to the default Kernel.Parameters(), according to read_from_file.
            likelihood_variance: The likelihood variance to use instead of file or default.
        Raises:
            IndexError: If a parameter is mis-shaped.
        """
        self._fold = fold
        self._X, self._Y = self._fold.X.to_numpy(dtype=FLOAT(), copy=True), self._fold.Y.to_numpy(dtype=FLOAT(), copy=True)
        self._N, self._M, self._L = self._fold.N, self._fold.M, self._fold.L
        super().__init__(self._fold.folder / name, is_read)
        self._likelihood = Likelihood(self, is_read) if likelihood_variance is None else Likelihood(self, is_read, variance=likelihood_variance)
        if is_read and kernel_parameters is None:
            KernelType = Kernel.TypeFromIdentifier(self.params.kernel[0, 0])
            self._kernel = KernelType(self._folder / self.KERNEL_FOLDER_NAME, is_read)
        else:
            if kernel_parameters is None:
                kernel_parameters = Kernel.Parameters()
            KernelType = Kernel.TypeFromParameters(kernel_parameters)
            self._kernel = KernelType(self._folder / self.KERNEL_FOLDER_NAME, is_read, **kernel_parameters.as_dict())
            self._parameters.replace(kernel=np.atleast_2d(KernelType.TYPE_IDENTIFIER)).write()
        self.broadcast_parameters(is_covariant, is_isotropic)


# noinspection PyPep8Naming
class MOGP(GPR):
    """ Implementation of a Gaussian Process."""

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        return {'maxiter': 5000, 'gtol': 1E-16}

    @property
    def implementation(self) -> Tuple[Any, ...]:
        if self._implementation is None:
            if self._likelihood.is_covariant:
                self._implementation = tuple(mf.models.MOGPR(data=(self._X, self._Y), kernel=kernel, mean_function=None,
                                                             noise_variance=self._likelihood.params.variance)
                                             for kernel in self._kernel.implementation)
            else:
                self._implementation = tuple(gf.models.GPR(data=(self._X, self._Y[:, [l]]), kernel=kernel, mean_function=None,
                                                           noise_variance=max(self._likelihood.params.variance[0, l], self._likelihood.VARIANCE_FLOOR))
                                             for l, kernel in enumerate(self._kernel.implementation))
        return self._implementation

    def optimize(self, method: str = 'L-BFGS-B', **kwargs) -> Dict[str, Any]:
        """ Optimize the MOGP hyper-parameters.

        Args:
            method: The optimization algorithm (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
            kwargs: A Dict of implementation-dependent optimizer options, following the format of GPR.OPTIONS.
                Options for the kernel should be passed as kernel={see kernel.OPTIONS for format}.
                Options for the likelihood should be passed as likelihood={see likelihood.OPTIONS for format}.
        """
        options = (self._read_options() if self._options_json.exists() else self.OPTIONS)
        kernel_options = self._kernel.optimize(**(options.pop('kernel', {}) | kwargs.pop('kernel', {})))
        likelihood_options = self._likelihood.optimize(**(options.pop('likelihood', {}) | kwargs.pop('likelihood', {})))
        options.update(kwargs)
        options.pop('result', None)
        opt = gf.optimizers.Scipy()
        options.update({'result': str(tuple(opt.minimize(closure=gp.training_loss, variables=gp.trainable_variables, method=method, options=options)
                                                  for gp in self._implementation)), 'kernel': kernel_options, 'likelihood': likelihood_options})
        self._write_options(options)
        if self._likelihood.is_covariant:
            self._likelihood.parameters = self._likelihood.parameters.replace(variance=self._implementation[0].likelihood.variance.value.numpy(),
                                                                              log_marginal=self._implementation[0].log_marginal_likelihood().numpy()
                                                                              ).write()
            self._kernel.parameters = self._kernel.parameters.replace(variance=self._implementation[0].kernel.variance.value.numpy(),
                                                                      lengthscales=tf.squeeze(self._implementation[0].kernel.lengthscales),
                                                                      ).write()
        else:
            self._likelihood.parameters = self._likelihood.parameters.replace(variance=tuple(gp.likelihood.variance.numpy() for gp in self._implementation),
                                                                              log_marginal=tuple(gp.log_marginal_likelihood() for gp in self._implementation)
                                                                              ).write()
            self._kernel.parameters = self._kernel.parameters.replace(variance=tuple(gp.kernel.variance.numpy() for gp in self._implementation),
                                                                      lengthscales=tuple(gp.kernel.lengthscales.numpy() for gp in self._implementation)
                                                                      ).write()
        return options

    def predict(self, X: NP.Matrix, y_instead_of_f: bool = True) -> Tuple[NP.Matrix, NP.Matrix]:
        X = X.astype(dtype=FLOAT())
        if self._likelihood.is_covariant:
            gp = self.implementation[0]
            results = gp.predict_y(X) if y_instead_of_f else gp.predict_f(X)
        else:
            results = tuple(gp.predict_y(X) if y_instead_of_f else gp.predict_f(X) for gp in self._implementation)
            results = tuple(np.transpose(result) for result in zip(*results))
            results = tuple(results[i][0] for i in range(len(results)))
        return np.atleast_2d(results[0]), np.atleast_2d(np.sqrt(results[1]))

    @property
    def X(self) -> TF.Matrix:
        """ The implementation training inputs as an (N,M) design matrix."""
        return self._implementation[0].data[0]

    @property
    def Y(self) -> TF.Matrix:
        """ The implementation training outputs as an (N,L) design matrix. """
        return self._implementation[0].data[1] if self._likelihood.is_covariant else tf.concat([gp.data[1] for gp in self._implementation], axis=1)

    @property
    def K_cho(self) -> TF.Tensor:
        if self._likelihood.is_covariant:
            gp = self._implementation[0]
            result = gp.likelihood.add_to(gp.KXX)
        else:
            result = []
            for gp in self._implementation:
                K = gp.kernel(self.X)
                K_diag = tf.linalg.diag_part(K)
                result.append(tf.linalg.set_diag(K, K_diag + tf.fill(tf.shape(K_diag), gp.likelihood.variance)))
            result = tf.stack(result)
        return tf.linalg.cholesky(result)

    @property
    def K_inv_Y(self) -> TF.Tensor:
        Y = tf.reshape(tf.transpose(self.Y), [-1, 1] if self._likelihood.is_covariant else tf.transpose(self.Y)[..., tf.newaxis])
        return tf.reshape(tf.linalg.cholesky_solve(self.K_cho, Y), [self._L, 1, self._N])

    def check_K_inv_Y(self, x: NP.Matrix) -> NP.Matrix:
        """ FOR TESTING PURPOSES ONLY. Should return 0 Vector (to within numerical error tolerance).

        Args:
            x: An (o, M) matrix of inputs.
        Returns: Should return zeros((Lo)) (to within numerical error tolerance).
        """
        predicted = self.predict(x)[0]
        o = predicted.shape[0]
        if self._likelihood.is_covariant:
            kernel = tf.reshape(self._implementation[0].kernel(x, self.X), [self._L, o, self._L, self._N])
            ein = 'loLN, iLN -> ol'
        else:
            kernel = tf.stack([gp.kernel(x, self.X) for gp in self._implementation], axis=0)
            ein = 'loN, liN -> ol'
        result = tf.einsum(ein, kernel, self.K_inv_Y)
        result -= predicted
        return tf.sqrt(tf.reduce_sum(result * result, axis=0)/o)

    # def __init__(self, name: str, fold: Fold, is_read: bool, is_covariant: bool, is_isotropic: bool,
    #              kernel_parameters: Optional[Kernel.Parameters] = None, likelihood_variance: NP.Matrix | None = None):
    #     """ MOGP Constructor. Calls __init__ to setup parameters, then checks dimensions.
    # 
    #     Args:
    #         name: The name of this MOGP.
    #         fold: The Fold housing this MOGP.
    #         is_read: If True, the MOGP.kernel.parameters and MOGP.parameters and are read from ``fold.folder/name``, otherwise defaults are used.
    #         is_covariant: Whether the outputs will be treated as independent.
    #         is_isotropic: Whether to restrict the kernel to be isotropic.
    #         kernel_parameters: A Kernel.Parameters to use for MOGP.kernel.parameters. If not None, this replaces the kernel specified by file/defaults.
    #             If None, the kernel is read from file, or set to the default Kernel.Parameters(), according to read_from_file.
    #         likelihood_variance: The likelihood variance to use instead of file or default.
    #     Raises:
    #         IndexError: If a parameter is mis-shaped.
    #     """
    #     super().__init__(name, fold, is_read, is_covariant, is_isotropic, kernel_parameters, likelihood_variance)
