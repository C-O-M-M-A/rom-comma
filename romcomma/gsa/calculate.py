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

""" Contains the calculation of a single coefficient of determination (closed Sobol index) without storing it."""

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.gpr.models import GPInterface


LOG2PI = tf.math.log(tf.constant(2 * np.pi, dtype=FLOAT()))
Triad = Dict[str, TF.Tensor]
TriadOfTriads = Dict[str, Triad]


def gaussian(mean: TF.Tensor, variance: TF.Tensor, is_variance_diagonal: bool,
             ordinate: TF.Tensor = tf.constant(0, dtype=FLOAT())) -> TF.Tensor:
    """ Computes the tensor gaussian probability density. Batch dimensions of ordinate, mean and variance are internally broadcast to match each other.

    Args:
        mean: Gaussian population mean.
        variance: The lower triangular Cholesky decomposition of the Gaussian population variance.
        is_variance_diagonal: True if variance is an m-vector
        ordinate: The ordinate (z-value) to calculate the Gaussian density for.
    Returns: The tensor Gaussian pdf, whose dimensions are the broadcast batch dimensions of ordinate, mean and variance.
    """
    ordinate = ordinate - mean if ordinate.shape else mean
    result = ordinate / variance if is_variance_diagonal else tf.linalg.triangular_solve(variance, ordinate, lower=True)
    result = tf.einsum('...o, ...o -> ...', result, result)
    result = result + tf.reduce_prod(variance) if is_variance_diagonal else tf.reduce_prod(tf.linalg.diag_part(variance))
    return tf.math.exp(-0.5 * (LOG2PI * ordinate.shape[-1] + result))


class ClosedIndexInterface(gf.Module):
    """ Interface encapsulating calculation of a single closed Sobol index."""

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        """ Calculation options. ``is_T_partial`` forces W[`m`][`M`] = W[`M`][`M`] = 0.

        Args:
            ms: The range of ms to calculate for.
            is_S_diagonal: If True, only the L-diagonal (variance) elements of S and V are calculated.
            is_T_calculated: If False, T and W return TF.NaN.
            is_T_diagonal: If True, only the S.shape-diagonal elements of T and W are calculated.
                In other words the variance of each element of S is calculated, but cross-covariances are not.
            is_T_partial: If True this effectively forces W[`m`][`M`] = W[`M`][`M`] = 0, as if the full ['M'] model is variance free.
        """
        return {'is_S_diagonal': False, 'is_T_calculated': True, 'is_T_diagonal': True, 'is_T_partial': False}

    @classmethod
    def Triad(cls) -> Dict[str, Union[Triad, TF.Tensor]]:
        """ Data structure for a Triad, populated with TF.NaNs."""
        return {'0': TF.NaN, 'm': TF.NaN, 'M': TF.NaN}

    @property
    def gp(self):
        return self._gp

    @property
    def S(self) -> TF.Tensor:
        """ The closed Sobol index tensor shaped (1,L) if ``is_S_diagonal`` else (L,L)."""
        return self._S['m']

    @property
    def T(self) -> TF.Tensor:
        """ The cross covariance of S, shaped (1,L,1,1) (if ``is_S_diagonal and is_T_diagonal``), (L,L,1,1) (if ``not is_S_diagonal and is_T_diagonal``)
        (1,L,1,L)  (if ``is_S_diagonal and not is_T_diagonal``), or (L,L,L,L)  (if not ``is_S_diagonal and not is_T_diagonal``). """
        return self._T['m']

    @property
    def V(self) -> Triad:
        """ The conditional variance Triad. Each element is shaped (L,) if ``is_S_diagonal`` else (L, L)."""
        return self._V

    @property
    def W(self) -> TriadOfTriads:
        """ The cross covariance of V, shaped (1,L,1,1) (if ``is_S_diagonal and is_T_diagonal``), (L,L,1,1) (if ``not is_S_diagonal and is_T_diagonal``)
        (1,L,1,L)  (if ``is_S_diagonal and not is_T_diagonal``), or (L,L,L,L)  (if not ``is_S_diagonal and not is_T_diagonal``). """
        return self._W

    @property
    def Theta(self) -> TF.Matrix:
        """ The input rotation matrix."""
        return tf.eye(self._M, dtype=FLOAT()) if self._Theta is None else self._Theta

    @Theta.setter
    def Theta(self, value: TF.Matrix):
        self._Theta = value
        self._calculate_without_marginalizing()
        self._m = self._M

    @property
    def m(self) -> TF.Tensor:
        """ The number of input dimensions for this ClosedIndex, 0 <= m <= self._M.

        Raises:
            AssertionError: Unless 0 < m <= self._M.
            TypeError: Unless dtype is INT().
        """
        return self._m

    @m.setter
    def m(self, value: int):
        assert 0 < value < self._M, f'm={value} is undoable.'
        if self._m != value:
            self._m = value
            self._marginalize()

    @abstractmethod
    def _marginalize(self):
        raise NotImplementedError

    @abstractmethod
    def _calculate_without_marginalizing(self):
        raise NotImplementedError

    def _Lambda2_(self):
        if self._gp.kernel.is_independent:
            self._Lambda2 = tf.einsum('LM,LM -> LM', self._lambda, self._lambda)[tf.newaxis, ...]
        else:
            self._Lambda2 = tf.einsum('LM,lM -> LlM', self._lambda, self._lambda)
        self._Lambda2 = tuple(self._Lambda2 + j for j in range(2))
        self._Lambda2 = {1: self._Lambda2, -1: tuple(value**(-1) for value in self._Lambda2)}
        self._pmF = tf.sqrt((2 * np.pi)**(self._M) * tf.reduce_prod(self._Lambda2[1][0], axis=-1)) * self._F

    def _ein_(self):
        Ls = 10
        Ms = 6
        Ns = 6
        self._einL = [chr(ord('A') + i) for i in range(Ls)]
        self._einM = [chr(ord('a') + i) for i in range(Ms)]
        self._einN = [chr(ord('a') + Ms + i) for i in range(Ns)]
        self._einLL = [self._einL[i] + self._einL[i + 1] for i in range(Ls // 2)]
        self._einLLL = [self._einL[i] + self._einL[i + 1] + self._einL[i + 2] for i in range(Ls // 3)]
        self._einLM = [[self._einL[i] + self._einM[j] for j in range(Ms//3)] for i in range(Ls//2)]
        self._einLLM = [[self._einLL[i] + self._einM[j] for j in range(Ms//3)] for i in range(Ls//2)]

    def __init__(self, gp: GPInterface, **kwargs: Any):
        """ Construct a ClosedIndex object.

        Args:
            gp: The gp to analyze.
            **kwargs: The calculation options to override OPTIONS.
        """
        super(ClosedIndexInterface, self).__init__()
        self._ein_()
        self._gp = gp
        self._options = self.OPTIONS | kwargs
        # Unwrap parameters
        self._L, self._M, self._N = self._gp.L, self._gp.M, self._gp.N
        self._E = tf.constant(self._gp.likelihood.params.variance, dtype=FLOAT())
        self._F = tf.constant(self._gp.kernel.params.variance, dtype=FLOAT())
        self._lambda = tf.constant(self._gp.kernel.params.lengthscales, dtype=FLOAT())
        self._Lambda2_()
        # Cache the training data kernel
        self._KNoisy_Cho = self._gp.KNoisy_Cho
        self._KNoisyInv_Y = self._gp.KNoisyInv_Y
        # Calculate the initial values
        self._Theta = None
        self._m = self._M
        self._calculate_without_marginalizing()
        self._marginalize()

        
class ClosedIndex(ClosedIndexInterface):
    """ Implements calculation of a single closed Sobol index."""

    def _marginalize(self):
        self._V_()
        if self._m == self._M:
            self._S = ClosedIndex.Triad()
            self._T = ClosedIndex.Triad()
            self._W = ClosedIndex.Triad()
            self._S['0'] = tf.zeros_like(self._V['0'])
            self._S['M'] = tf.ones_like(self._V['M'])
        self._S['m'] = self._V['m'] / self._V['M']
        if self._options['is_T_calculated']:
            if self._m == self._M:
                self._VV = ClosedIndex.Triad()
                self._VV['m'] = ClosedIndex.Triad()
                if self._options['is_T_diagonal']:
                    self._VV['M'] = tf.einsum('Ll,Ll -> Ll', self._V['M'], self._V['M'])[..., tf.newaxis, tf.newaxis]
                else:
                    self._VV['M'] = tf.einsum('Ll,Jj -> LlJj', self._V['M'], self._V['M'])
                self._T['0'] = tf.zeros_like(self._VV['M'])
                self._T['M'] = tf.zeros_like(self._VV['M'])
            self._W_()
            self._T['m'] = self._W['m']['m'] / self._VV['M']
            if not self._options['is_T_partial']:
                self._T['m'] = self._T['m'] + (self._VV['m']['m'] / self._VV['M']) * (self._W['M'] / self._VV['M']
                                                                                      - 2 * self._W['m']['M'] / (self._VV['m']['M']))

    def _V_(self):
        self._mu1mu1_()
        if self._m == self._M:
            self._V = ClosedIndex.Triad()
            self._V['0'] = self._mu1mu1['0']
            self._V['M'] = self._mu1mu1['M'] - self._V['0']
        self._V['m'] = self._mu1mu1['m'] - self._V['0']

    def _mu1mu1_(self):
        shape = (1, self._L) if self._options['is_S_diagonal'] else (self._L, self._L)
        if self._m == self._M:
            self._mu1mu1 = ClosedIndex.Triad()
            self._mu1mu1['0'] = 0.5 * tf.ones(shape, dtype=FLOAT())
            self._mu1mu1['M'] = tf.ones(shape, dtype=FLOAT())
        self._mu1mu1['m'] = tf.ones(shape, dtype=FLOAT())

    def _W_(self):
        self._mu1mu2mu1_()
        if self._m == self._M:
            self._W['m'] = ClosedIndex.Triad()
            if not self._options['is_T_partial']:
                self._W['M'] = 4 * (self._mu1mu2mu1['0'] - 2 * self._mu1mu2mu1['M']['0'] + self._mu1mu2mu1['M']['M'])
        self._W['m']['m'] = 4 * (self._mu1mu2mu1['0'] - 2 * self._mu1mu2mu1['m']['0'] + self._mu1mu2mu1['m']['m'])
        if not self._options['is_T_partial']:
            if self._options['is_T_diagonal']:
                self._VV['m']['m'] = tf.einsum('Ll,Ll -> Ll', self._V['m'], self._V['m'])[..., tf.newaxis, tf.newaxis]
                self._VV['m']['M'] = tf.einsum('Ll,Ll -> Ll', self._V['m'], self._V['M'])[..., tf.newaxis, tf.newaxis]
            else:
                self._VV['m']['m'] = tf.einsum('Ll,Jj -> LlJj', self._V['m'], self._V['m'])
                self._VV['m']['M'] = tf.einsum('Ll,Jj -> LlJj', self._V['m'], self._V['M'])
            self._W['m']['M'] = 4 * (self._mu1mu2mu1['0'] - self._mu1mu2mu1['m']['0'] - self._mu1mu2mu1['M']['0'] + self._mu1mu2mu1['m']['M'])

    def _mu1mu2mu1_(self):
        if self._m == self._M:
            self._mu1mu2mu1 = ClosedIndex.Triad()
            self._mu1mu2mu1['m'] = ClosedIndex.Triad()
            self._mu1mu2mu1['0'] = tf.zeros_like(self._VV['M'])
            if not self._options['is_T_partial']:
                self._mu1mu2mu1['M'] = ClosedIndex.Triad()
                self._mu1mu2mu1['M']['0'] = tf.ones_like(self._VV['M'])
                self._mu1mu2mu1['M']['M'] = tf.ones_like(self._VV['M'])
        self._mu1mu2mu1['m']['0'] = tf.ones_like(self._VV['M'])
        self._mu1mu2mu1['m']['m'] = tf.ones_like(self._VV['M'])
        if not self._options['is_T_partial']:
            self._mu1mu2mu1['m']['M'] = tf.ones_like(self._VV['M'])

    def _calculate_without_marginalizing(self):
        pass
