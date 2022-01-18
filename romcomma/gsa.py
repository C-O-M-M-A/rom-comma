#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2022 Robert A. Milton. All rights reserved.
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

""" Contains Global Sensitivity Analysis tools."""

from romcomma._common_definitions import *
from romcomma.base import Model, Parameters
from romcomma.gpr import GPInterface

LOG2PI = tf.math.log(tf.constant(2 * np.pi, dtype=FLOAT()))

def gaussian(mean: TF.Tensor, variance: TF.Tensor, is_variance_diagonal: bool,
             ordinate: TF.Tensor = tf.constant(0, dtype=FLOAT())) -> TF.Tensor:
    """ Computes the gaussian probability density.

    Args:
        mean: Gaussian population mean.
        variance: The lower triangular Cholesky decomposition of the Gaussian population variance.
        is_variance_diagonal: True if variance is an m-vector
        ordinate: The ordinate (z-value) to calculate the Gaussian density for.
    Returns:
    """
    ordinate = ordinate - mean if ordinate.shape else mean
    result = ordinate / variance if is_variance_diagonal else tf.linalg.triangular_solve(variance, ordinate, lower=True)
    result = tf.reduce_sum(tf.square(result), axis=-1)
    result = result + tf.reduce_prod(variance) if is_variance_diagonal else tf.reduce_prod(tf.linalg.diag_part(variance))
    return tf.math.exp(-0.5 * (LOG2PI * ordinate.shape[-1] + result))


class GSAInterface(Model, gf.Module):
    """ Interface encapsulating a Global Sensitivity Analysis."""

    class Parameters(Parameters):
        """ The Parameters set of a GSA."""

        @classmethod
        @property
        def Values(cls) -> Type[NamedTuple]:
            """ The NamedTuple underpinning this Parameters set."""

            class Values(NamedTuple):
                """ The parameters set of a GSA.

                Attributes:
                    Theta (NP.Matrix): An (M,M) rotation matrix.
                    result (NP.Matrix): An (L, L, M) tensor result, flattened into an (L*L, M) matrix.
                    covariance (NP.Matrix): An (L, L, L, L, M, M) tensor covariance of result, flattened into an (L*L*L*L, M*M) matrix.
                """
                Theta: NP.Matrix = np.atleast_2d(0.0)
                result: NP.Matrix = np.atleast_2d(0.0)
                covariance: NP.Matrix = np.atleast_2d(0.0)

            return Values

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        """ Calculation options. Is covariance partial forces W[m][M] = W[M][M] = 0 """
        return {'is_S_diagonal': True, 'is_T_calculated': True, 'is_T_diagonal': True, 'is_T_partial': True}

    @classmethod
    @property
    def NaN(cls) -> TF.Tensor:
        return tf.constant(np.NaN, dtype=FLOAT())

    @classmethod
    @property
    def Triad(cls) -> Dict[str, TF.Tensor]:
        """ Data structure for a Triad, populated with NaNs."""
        return {'0': GSA.NaN, 'm': GSA.NaN, 'M': GSA.NaN}

    @property
    def S(self) -> TF.Tensor:
        """ The closed Sobol index tensor shaped (L,) """
        return self._S
    
    @property
    def Theta(self) -> TF.Matrix:
        """ The input rotation matrix."""
        return tf.cond(tf.reduce_any(tf.math.is_nan(self._Theta)), tf.eye(self.M, dtype=FLOAT()), self._Theta)

    @Theta.setter
    def Theta(self, value: TF.Matrix):
        self._Theta = value
        self._calculate_without_marginalizing()

    @property
    def m(self) -> TF.Tensor:
        """ The number of input dimensions for this GSA, 0 <= m <= self.M.

        Raises:
            AssertionError: Unless 0 < m <= self.M.
            TypeError: Unless dtype is INT()
        """
        return self._m

    @m.setter
    def m(self, value: TF.Tensor):
        tf.assert_greater(value, 0, f'm={value} must be between 0 and M+1={self.M+1}.')
        tf.assert_less(value, self.M + 1, f'm={value} must be between 0 and M+1={self.M+1}.')
        self._m = tf.cond(value == self._m, value, self._marginalize(value))

    @property
    def gp(self):
        return self._gp

    @property
    def L(self):
        return self._gp.L

    @property
    def M(self):
        return self._gp.M

    @property
    def N(self):
        return self._gp.N

    @abstractmethod
    def _marginalize_M0(self):
        """ Calculate the closed Sobol index tensors S['M'], S.['0'], and, optionally, their cross covariances T['M'], T['0']."""
        self._m = self.M

    @abstractmethod
    def _marginalize(self, m: int) -> int:
        """ Calculate the closed Sobol index tensor S['m'] of the first m input dimensions, and, optionally, its cross covariance T['m'].

        Args:
            m: The number of input dimensions for this GSA, 0 <= m <= self.M.
        Returns: m
        """
        self._m = m
        return m

    @abstractmethod
    def _calculate_without_marginalizing_M0(self):
        """ Calculate all quantities up to marginalization. Essentially, this calculates all the means and variances of all Gaussians behind the GSA,
        but stops short of calculating the (marginalized) Gaussians."""
        raise NotImplementedError

    @abstractmethod
    def _calculate_without_marginalizing(self):
        """ Calculate all quantities up to marginalization. Essentially, this calculates all the means and variances of all Gaussians behind the GSA,
        but stops short of calculating the (marginalized) Gaussians."""
        raise NotImplementedError

    def _calculate_M0(self):
        """ Calculate all quantities for m=M and m=0, setting Theta=None and m=0 for this GSA."""
        self._Theta = None
        self._calculate_without_marginalizing_M0()
        self._marginalize_M0()
        self._m = 0

    def _Lambda2_(self) -> Dict[int, Tuple[TF.Tensor]]:
        Lambda2 = self._lambda**2 if self._gp.kernel.is_independent else tf.einsum('LM,lM -> LlM', self._lambda, self._lambda)
        Lambda2 = tuple(Lambda2 + j for j in range(2))
        return {1: Lambda2, -1: tuple(value**(-1) for value in Lambda2)}

    def __init__(self, name: str, gp: GPInterface, **kwargs: Any):
        """ Construct a GSA object.

        Args:
            name: The name of the GSA.
            gp: The gp to analyze.
            **kwargs: Set calculation options, by updating OPTIONS.
        """
        super(Model, self).__init__(name=name)
        self._gp = gp
        self._folder = self._gp.folder / 'gsa' / name
        self.options = self.OPTIONS | kwargs
        # if Theta is None and self._folder.exists():
        #     super().__init__(folder=self._folder, read_parameters=True)
        # else:
        #     Theta = np.eye(self.M, dtype=FLOAT()) if Theta is None else Theta
        #     super().__init__(folder=self._folder, read_parameters=False, Theta=Theta)
        # Theta = tf.constant(self.params.Theta, dtype=FLOAT())

        # Unwrap parameters
        self._L, self._N = self._gp.L, self._gp.N
        self._E = tf.constant(self._gp.likelihood.params.variance, dtype=FLOAT())
        self._F = tf.constant(self._gp.kernel.params.variance, dtype=FLOAT())
        self._lambda = tf.constant(self._gp.kernel.params.lengthscales, dtype=FLOAT())
        self._Lambda2 = self._Lambda2_()

        # Cache the training data kernel
        self._KNoisy_Cho = self._gp.KNoisy_Cho
        self._KNoisyInv_Y = self._gp.KNoisyInv_Y

        # Calculate the initial values
        self._calculate_M0()

        
class GSA(GSAInterface):
    """ Implementation of Global Sensitivity Analysis."""

    def _marginalize_M0(self):
        super(GSA, self)._marginalize_M0()
        self._V_M0()
        self._S = GSA.Triad
        self._S['0'] = tf.zeros_like(self._V['M'])
        self._S['M'] = tf.ones_like(self._V['M'])
        self._W_M0()
        self._T = GSA.Triad
        self._T['0'] = tf.zeros_like(self._W['M']['M'])
        self._T['M'] = tf.zeros_like(self._W['M']['M'])

    def _V_M0(self):
        self._V = GSA.Triad
        self._V['0'] = tf.cond(self.options['is_S_diagonal'], tf.zeros(shape=(self.L,), dtype=FLOAT()),
                                  tf.ones(shape=(self.L, self.L), dtype=FLOAT()))
        self._V['M'] = tf.cond(self.options['is_S_diagonal'], tf.ones(shape=(self.L,), dtype=FLOAT()),
                                  tf.ones(shape=(self.L, self.L), dtype=FLOAT())) - self._V['0']

    def _W_M0(self):
        self._W = GSA.Triad
        self._W['M'] = 0.5 * tf.cond(self.options['is_T_calculated'], tf.ones(shape=(self.L, self.L), dtype=FLOAT()), GSA.NaN)

    def _marginalize(self, m: int) -> int:
        super(GSA, self)._marginalize(m)
        self._V_m()
        self._S['m'] = self._V['m'] / self._V['M']
        self._W_m()
        self._T['m'] = (tf.square(self._V['m']) / tf.square(self._V['M'])) * (self._W['m']['m'] / tf.square(self._V['m'])
                                                                              + self._W['M'] / tf.square(self._V['M'])
                                                                              - 2 * self._W['m']['M'] / (self._V['m'] * self._V['M']))
        return m

    def _V_m(self):
        self._V['m'] = 0.5 * tf.cond(self.options['is_S_diagonal'], tf.ones(shape=(self.L,), dtype=FLOAT()),
                                     tf.ones(shape=(self.L, self.L), dtype=FLOAT()))

    def _W_m(self):
        self._W['m'] = GSA.Triad
        self._W['m']['m'] = tf.cond(self.options['is_T_calculated'], tf.ones(shape=(self.L, self.L), dtype=FLOAT()), GSA.NaN)
        self._W['m']['M'] = tf.cond(self.options['is_T_calculated'], tf.ones(shape=(self.L, self.L), dtype=FLOAT()), GSA.NaN)

    def calculate_without_marginalizing_M0(self):
        pass

    def calculate_without_marginalizing(self):
        raise NotImplementedError
