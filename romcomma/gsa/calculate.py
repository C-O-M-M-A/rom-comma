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


class Gaussian:
    """ Encapsulates the calculation of a Gaussian pdf. Not instantiatable."""

    @staticmethod
    def pdf_with_det(mean: TF.Tensor, variance_cho: TF.Tensor, is_variance_diagonal: bool,
                     ordinate: TF.Tensor = tf.constant(0, dtype=FLOAT()), LBunch: int = 2) -> Tuple[TF.Tensor, TF.Tensor]:
        """ Computes the tensor gaussian probability density. Batch dimensions of ordinate, mean and variance are internally broadcast to match each other.

        Args:
            mean: Gaussian population mean. Should be of adequate rank to broadcast Ls.
            variance_cho: The lower triangular Cholesky decomposition of the Gaussian population variance. Is automatically broadcast to embrace Ns
            is_variance_diagonal: True if variance is an M-vector
            ordinate: The ordinate (z-value) to calculate the Gaussian density for. Should be of adequate rank to broadcast Ls. If not supplied, 0 is assumed.
            LBunch: The number of consecutive output (L) dimensions to count before inserting an N for broadcasting. Usually 2, sometimes 3.
        Returns: The tensor Gaussian pdf, and the determinant of variance_cho.
        """
        # Broadcast ordinate - mean.
        if ordinate.shape == mean.shape:
            shape = ordinate.shape.as_list()
            fill = [1, ] * (len(shape) - 1)
            ordinate = tf.reshape(ordinate, shape[:-1] + fill + [shape[-1]])
            mean = tf.reshape(mean, fill + shape)
        ordinate = ordinate - mean
        # Broadcast variance_cho
        insertions = (tf.rank(variance_cho) - (1 if is_variance_diagonal else 2))
        insertions -= insertions % LBunch
        for axis in range(insertions, 0, -LBunch):
            variance_cho = tf.expand_dims(variance_cho, axis=axis)
        # Calculate the Gaussian pdf.
        result = ordinate / variance_cho if is_variance_diagonal else tf.linalg.triangular_solve(variance_cho, ordinate, lower=True)
        result = tf.einsum('...o, ...o -> ...', result, result)
        det_cho = tf.reduce_prod(variance_cho if is_variance_diagonal else tf.linalg.diag_part(variance_cho), axis=-1)
        return tf.math.exp(-0.5 * (result + LOG2PI * ordinate.shape[-1])) / det_cho, det_cho

    @staticmethod
    def pdf(mean: TF.Tensor, variance_cho: TF.Tensor, is_variance_diagonal: bool,
                     ordinate: TF.Tensor = tf.constant(0, dtype=FLOAT()), LBunch: int = 2) -> TF.Tensor:
        """ Computes the tensor gaussian probability density. Batch dimensions of ordinate, mean and variance are internally broadcast to match each other.

        Args:
            mean: Gaussian population mean. Should be of adequate rank to broadcast Ls.
            variance_cho: The lower triangular Cholesky decomposition of the Gaussian population variance. Is automatically broadcast to embrace Ns
            is_variance_diagonal: True if variance is an M-vector
            ordinate: The ordinate (z-value) to calculate the Gaussian density for. Should be of adequate rank to broadcast Ls. If not supplied, 0 is assumed.
            LBunch: The number of consecutive output (L) dimensions to count before inserting an N for broadcasting. Usually 2, sometimes 3.
        Returns: The tensor Gaussian pdf.
        """
        # Broadcast ordinate - mean.
        return Gaussian.pdf_with_det(mean, variance_cho, is_variance_diagonal, ordinate, LBunch)[0]


class Dagger:
    """ Implements the dagger operation. Not instantiatable."""

    @staticmethod
    def apply(tensor: TF.Tensor) -> TF.Tensor:
        """ Dagger reshape tensor.

        Args:
            tensor: Must be shaped [L, L, N] * J
        Returns: tensor reshaped to [L, LN] * J
        """
        shape = tensor.shape.as_list()
        for i in range(len(shape) - 1, 0, -3):
            shape[i - 1] *= shape[i]
            del shape[i]
        return tf.reshape(tensor, shape)

    @staticmethod
    def undo(tensor: TF.Tensor, is_L_diagonal: bool) -> TF.Tensor:
        """ Undagger reshape tensor.

        Args:
            tensor: Must be shaped [L, LN] * J or [L, N] * J
            is_L_diagonal: True inserts 1, False inserts L.
        Returns: tensor reshaped to [L, L, N] * J or [L, 1, N * J]
        """
        shape = tensor.shape.as_list()
        insert = 1 if is_L_diagonal else shape[-2]
        for i in range(len(shape) - 1, 0, -2):
            shape[i] /= shape[i-1]
            shape = shape.insert(i, insert)
        return tf.reshape(tensor, shape)


class ClosedIndexInterface(gf.Module):
    """ Interface encapsulating calculation of a single closed Sobol index."""

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        """ Calculation options. ``is_T_partial`` forces W[`m`][`M`] = W[`M`][`M`] = 0.

        Returns:
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
        return self._Theta

    @Theta.setter
    def Theta(self, value: TF.Matrix):
        if not tf.reduce_all(tf.equal(value, self._Theta)):
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
            self._Lambda2 = tf.expand_dims(tf.einsum('LM, LM -> LM', self._lambda, self._lambda), axis=1)
        else:
            self._Lambda2 = tf.einsum('LM, lM -> LlM', self._lambda, self._lambda)
        self._Lambda2 = tuple(self._Lambda2 + j for j in range(3))
        self._Lambda2 = {1: self._Lambda2, -1: tuple(value**(-1) for value in self._Lambda2)}

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
        self._F = tf.transpose(tf.constant(self._gp.kernel.params.variance, dtype=FLOAT()))     # To convert (1,L) to (L,1)
        self._lambda = tf.constant(self._gp.kernel.params.lengthscales, dtype=FLOAT())
        self._Lambda2_()
        self._pmF = tf.sqrt((2 * np.pi)**(self._M) * tf.reduce_prod(self._Lambda2[1][0], axis=-1)) * self._F
        # Cache the training data kernel
        self._KNoisy_Cho = self._gp.KNoisy_Cho
        self._KNoisyInv_Y = self._gp.KNoisyInv_Y
        # Calculate selected constants
        self._g0_dag = self._pmF[..., tf.newaxis] * Gaussian.pdf(mean=self._gp.X[tf.newaxis, tf.newaxis, ...],
                                                                 variance_cho=tf.sqrt(self._Lambda2[1][1]), is_variance_diagonal=True)
        self._g0_dag = Dagger.apply(self._g0_dag)
        self._KYg0 = self._g0_dag * tf.reshape(self._KNoisyInv_Y, [self._L, self._N] if self._gp.kernel.is_independent else [1,-1])
        if self._options['is_T_calculated']:
            self._g0_2 = tf.einsum('LA, la -> LAla', self._g0_dag, self._g0_dag)
        # Calculate the initial results
        self._Theta = tf.eye(self._M, dtype=FLOAT())
        self._m = self._M
        self._calculate_without_marginalizing()
        self._marginalize()

        
class OldClosedIndex(ClosedIndexInterface):
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
                self._T['0'] = tf.zeros(self._T_shape, FLOAT())
                self._T['M'] = tf.zeros(self._T_shape, FLOAT())
            self._W_()
            self._T['m'] = self._W['m']['m'] / self._VV['M']
            if not self._options['is_T_partial']:
                self._T['m'] = self._T['m'] + (self._VV['m']['m'] / self._VV['M']) * (self._W['M'] / self._VV['M']
                                                                                      - 2 * self._W['m']['M'] / (self._VV['m']['M']))

    def _V_(self):
        self._S_shape = (1, self._L) if self._options['is_S_diagonal'] else (self._L, self._L)
        self._T_shape = self._S_shape * 2 if self._options['is_T_diagonal'] else self._S_shape + (1, 1)
        self._mu1mu1_()
        if self._m == self._M:
            self._V = ClosedIndex.Triad()
            self._V['0'] = self._mu1mu1['0']
            self._V['M'] = self._mu1mu1['M'] - self._V['0']
        self._V['m'] = self._mu1mu1['m'] - self._V['0']

    def _mu1mu1_(self):
        if self._m == self._M:
            self._mu1mu1 = ClosedIndex.Triad()
            self._mu1mu1['0'] = 0.5 * tf.ones(self._S_shape, dtype=FLOAT())
            self._mu1mu1['M'] = tf.ones(self._S_shape, dtype=FLOAT())
        self._mu1mu1['m'] = tf.ones(self._S_shape, dtype=FLOAT())

    def _W_(self):
        self._mu1mu2mu1_()
        if self._m == self._M:
            self._VV = ClosedIndex.Triad()
            self._VV['m'] = ClosedIndex.Triad()
            if self._options['is_T_diagonal']:
                self._VV['M'] = tf.einsum('Ll, Ll -> Ll', self._V['M'], self._V['M'])[..., tf.newaxis, tf.newaxis]
            else:
                self._VV['M'] = tf.einsum('Ll, Jj -> LlJj', self._V['M'], self._V['M'])
            if not self._options['is_T_partial']:
                self._W['M'] = 4 * (self._mu1mu2mu1['0'] - 2 * self._mu1mu2mu1['M']['0'] + self._mu1mu2mu1['M']['M'])
            self._W['m'] = ClosedIndex.Triad()
        self._W['m']['m'] = 4 * (self._mu1mu2mu1['0'] - 2 * self._mu1mu2mu1['m']['0'] + self._mu1mu2mu1['m']['m'])
        if not self._options['is_T_partial']:
            if self._options['is_T_diagonal']:
                self._VV['m']['m'] = tf.einsum('Ll, Ll -> Ll', self._V['m'], self._V['m'])[..., tf.newaxis, tf.newaxis]
                self._VV['m']['M'] = tf.einsum('Ll, Ll -> Ll', self._V['m'], self._V['M'])[..., tf.newaxis, tf.newaxis]
            else:
                self._VV['m']['m'] = tf.einsum('Ll, Jj -> LlJj', self._V['m'], self._V['m'])
                self._VV['m']['M'] = tf.einsum('Ll, Jj -> LlJj', self._V['m'], self._V['M'])
            self._W['m']['M'] = 4 * (self._mu1mu2mu1['0'] - self._mu1mu2mu1['m']['0'] - self._mu1mu2mu1['M']['0'] + self._mu1mu2mu1['m']['M'])

    def _mu1mu2mu1_(self):
        if self._m == self._M:
            self._mu1mu2mu1 = ClosedIndex.Triad()
            self._mu1mu2mu1['m'] = ClosedIndex.Triad()
            self._mu1mu2mu1['0'] = tf.zeros(self._T_shape, FLOAT())
            if not self._options['is_T_partial']:
                self._mu1mu2mu1['M'] = ClosedIndex.Triad()
                self._mu1mu2mu1['M']['0'] = tf.ones(self._T_shape, FLOAT())
                self._mu1mu2mu1['M']['M'] = tf.ones(self._T_shape, FLOAT())
        self._mu1mu2mu1['m']['0'] = tf.ones(self._T_shape, FLOAT())
        self._mu1mu2mu1['m']['m'] = tf.ones(self._T_shape, FLOAT())
        if not self._options['is_T_partial']:
            self._mu1mu2mu1['m']['M'] = tf.ones(self._T_shape, FLOAT())

    def _matrix_inverse(self, tensor: TF.Tensor, I: tf.Tensor = None) -> TF.Tensor:
        """ Invert the inner matrix of an (L,L,M,M) or (L,L,L,L,M,M) Tensor.

        Args:
            tensor: A tensor whose shape matches identity.
            I: Supply the (L,L,M,M) identity matrix, otherwise the (L,L,L,L,M,M) identity matrix is used.
        Returns: The inner matrix inverse of tensor.
        """
        if I is None:
            I = tf.eye(self._M, batch_shape=[1, 1, 1, 1], dtype=FLOAT())
            ein = 'IiLlmM, IiLlmJ -> IiLlMJ'
        else:
            ein = 'LlmM, LlmJ -> LlMJ'
        result = tf.linalg.cholesky(tensor)
        result = tf.linalg.triangular_solve(result, I)
        return tf.einsum(ein , result, result)

    def _calculate_without_marginalizing(self, Theta: TF.Matrix):
        I = tf.eye(self._M, batch_shape=[1, 1], dtype=FLOAT())
        # First Moments
        G = tf.einsum('Mm, Llm, Nm -> LlNM', Theta, self._Lambda2[-1][1], self._gp.X)
        Phi = tf.einsum('Mm, Llm, Jm -> LlMJ', Theta, self._Lambda2[-1][1], Theta)
        Gamma = I - Phi
        # Expected Value
        Sigma = tf.reshape(Gamma, [self._L, self._L, 1, 1, self._M, self._M]) + Gamma[tf.newaxis, tf.newaxis,...]
        Psi = Sigma - tf.einsum('IiMm, LlmJ -> IiLlMJ', Gamma, Gamma)
        SigmaPsi = tf.einsum('IiLlMm, IiLlmJ -> IiLlMJ', Sigma, Psi)
        Gamma_reshape = tf.expand_dims(Gamma, 2)
        SigmaG = tf.einsum('IinMm, LlNm -> IinLlNM', Gamma_reshape, G) + tf.einsum('IinMm, LlNm -> LlNIinM', Gamma_reshape, G)
        # Second Moments
        Upsilon = tf.einsum('Mm, Llm, Llm, Jm -> LlMJ', Theta, self._Lambda2[1][1], self._Lambda2[-1][2], Theta)
        Gamma_inv = self._matrix_inverse(Gamma, I)
        print(tf.einsum('LlMm, LlmJ -> LlMJ', Gamma_inv, Gamma))
        Upsilon_inv = self._matrix_inverse(Upsilon, I)
        print(tf.einsum('LlMm, LlmJ -> LlMJ', Upsilon_inv, Upsilon))
        Pi = self._matrix_inverse(tf.einsum('LlMm, Llmj, LljJ -> LlMJ', Phi, Gamma_inv, Phi) + Upsilon_inv, I)
        # Variance
        sqrt_1_Upsilon = tf.linalg.band_part(tf.linalg.cholesky(I - Upsilon), -1, 0)

    def _calculate_without_marginalizing_check(self):
        pass


class ClosedIndex(gf.Module):
    """ Calculates closed Sobol Indices."""

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        """ Calculation options. ``is_T_partial`` forces W[`m`][`M`] = W[`M`][`M`] = 0.

        Returns:
            ms: The range of ms to calculate for.
            is_S_diagonal: If True, only the L-diagonal (variance) elements of S and V are calculated.
            is_T_calculated: If False, T and W return TF.NaN.
            is_T_diagonal: If True, only the S.shape-diagonal elements of T and W are calculated.
                In other words the variance of each element of S is calculated, but cross-covariances are not.
            is_T_partial: If True this effectively forces W[`m`][`M`] = W[`M`][`M`] = 0, as if the full ['M'] model is variance free.
        """
        return {'is_S_diagonal': False, 'is_T_calculated': True, 'is_T_diagonal': True, 'is_T_partial': False}

    @property
    def gp(self):
        return self._gp

    @property
    def V0(self) -> TF.Tensor:
        return self._V0

    def marginalize(self, m: TF.Tensor) -> Dict[str, TF.Tensor]:
        """

        Args:
            m:

        Returns:

        """
        raise NotImplementedError


    def _calculate(self):
        """ Calculate all required quantities for m=M. """
        # First Moments
        self._G = tf.einsum('lLM, NM -> lLNM', self._Lambda2[-1][1], self._gp.X)
        self._Phi = self._Lambda2[-1][1]
        self._Gamma = 1 - self._Phi
        # Second Moments
        self._Upsilon = self._Lambda2[1][1] * self._Lambda2[-1][2]
        self._Pi = self._Gamma
        # Expected Value
        Gamma_reshape = tf.expand_dims(self._Gamma, axis=2)
        self._Sigma = tf.expand_dims(Gamma_reshape, axis=2) + self._Gamma[tf.newaxis, tf.newaxis, ...]
        self._Psi = self._Sigma - tf.einsum('liM, jkM -> lijkM', self._Gamma, self._Gamma)
        self._SigmaPsi = tf.einsum('lijkM, lijkM -> lijkM', self._Sigma, self._Psi)
        self._SigmaG = tf.einsum('jknM, liNM -> liNjknM', Gamma_reshape, self._G) + tf.einsum('liNM, jknM -> liNjknM', Gamma_reshape, self._G)
        Sigma_pdf, Sigma_det = Gaussian.pdf_with_det(mean=self._G, ordinate=self._G, variance_cho=tf.sqrt(self._Sigma), is_variance_diagonal=True, LBunch=2)
        SigmaPsi_pdf, SigmaPsi_det = Gaussian.pdf_with_det(mean=self._SigmaG, variance_cho=tf.sqrt(self._SigmaPsi), is_variance_diagonal=True, LBunch=2)
        self._H = (Sigma_pdf / SigmaPsi_pdf) * (Sigma_det / SigmaPsi_det)**2
        self._V0 = tf.einsum('l, i -> li', self._KYg0_summed, self._KYg0_summed)
        # Variance
        self._sqrt_1_Upsilon = tf.sqrt(self._Lambda2[-1][2])  # = sqrt(1 - Upsilon)
        self._Omega = tf.einsum('kJM, ijM -> ijkJM', self._Phi, self._Phi)
        self._B = (tf.einsum('kJM, ijM, kJM -> ijkJM', self._Phi, self._Pi, self._Phi)
                   + tf.einsum('kJM, kJM -> kJM', self._Gamma, self._Phi)[tf.newaxis, tf.newaxis, ...])
        Gamma_reshape = tf.expand_dims(Gamma_reshape, axis=1)
        self._C = Gamma_reshape / (Gamma_reshape + tf.einsum('lLM, ijM -> liLjM', self._Phi, self._Upsilon))
        self._C = tf.einsum('ijM, liLjM -> liLjM', self._Upsilon, self._C)
        self._D = (tf.einsum('kKM, jJM, kKM -> jkJKM', self._Phi, self._Gamma, self._Phi)
                   + tf.expand_dims(tf.expand_dims(tf.einsum('kKM, kKM -> kKM', self._Gamma, self._Phi), axis=1), axis=0))

        self._mu_phi_mu['00'] = tf.einsum('ij, l, k -> lijk', self._mu_phi_mu['pre-factor'], self._KYg0_summed, self._KYg0_summed)
        sqrt_1_Upsilon_pdf = self._sqrt_1_Upsilon_pdf(True, self._sqrt_1_Upsilon, self._G, self._Phi)
        self._mu_phi_mu['M0'] = tf.einsum('ij, lLN, liLNj, k -> lijk', self._mu_phi_mu['pre-factor'], self._KYg0, sqrt_1_Upsilon_pdf, self._KYg0_summed)
        g_factor = self._KYg0 / Gaussian.pdf(self._G, tf.sqrt(self._Phi), True)
        self._mu_phi_mu['MM'] = tf.einsum('kJn, liLNjkJn -> liLNjk', g_factor,
                                          self._Omega_pdf(True, self._B, self._C, self._G, self._Omega, 1 / self._Gamma))
        self._mu_phi_mu['MM'] = tf.einsum('ij, lLN, liLNj, liLNjk -> lijk',
                                          self._mu_phi_mu['pre-factor'], self._KYg0, sqrt_1_Upsilon_pdf, self._mu_phi_mu['MM'])

    def _Omega_pdf(self, is_diagonal: bool, B: TF.Tensor, C: TF.Tensor, G: TF.Tensor, Omega: TF.Tensor, Gamma_inv: TF.Tensor) -> TF.Tensor:
        # The assumption here is that m >= m'. This is important
        ein = 'ijkJM, liLjM, lLM, lLNM -> liLNjkJM' if is_diagonal else 'ijkJMa, liLjab, lLbc, lLNc -> liLNjkJM'
        mean = tf.einsum(ein, Omega[..., :G.shape[-1]], C, Gamma_inv, G)
        mean = tf.expand_dims(mean, axis=7) - G[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]
        ein = 'ijkJM, liLjM, ijkJM -> liLjkJM' if is_diagonal else 'ijkJMa, liLjab, ijkJmb -> liLjkJMm'
        variance = tf.einsum(ein, Omega, C, Omega) + tf.expand_dims(tf.expand_dims(B, axis=1), axis=0)
        variance = tf.sqrt(variance) if is_diagonal else tf.linalg.cholesky(variance)
        return Gaussian.pdf(mean, variance, is_diagonal, LBunch=3)

    def _sqrt_1_Upsilon_pdf(self, is_diagonal: bool, sqrt_1_Upsilon: TF.Tensor, G: TF.Tensor, Phi: TF.Tensor) -> TF.Tensor:
        ein = 'ijM, lLNM -> liLNjM' if is_diagonal else 'ijMm, lLNm -> liLNjM'
        mean = tf.einsum(ein, sqrt_1_Upsilon, G)
        ein = 'ijM, lLM, ijM -> liLjM' if is_diagonal else 'ijMm, lLma, ijAa -> liLjMA'
        variance = tf.einsum(ein, sqrt_1_Upsilon, Phi, sqrt_1_Upsilon)
        variance = tf.sqrt(variance) if is_diagonal else tf.linalg.cholesky(variance)
        return Gaussian.pdf(mean, variance, is_diagonal, LBunch=3)

    def _calc_Lambda2_(self):
        if self._gp.kernel.is_independent:
            self._Lambda2 = tf.expand_dims(tf.einsum('LM, LM -> LM', self._lambda, self._lambda), axis=1)
        else:
            self._Lambda2 = tf.einsum('LM, lM -> LlM', self._lambda, self._lambda)
        self._Lambda2 = tuple(self._Lambda2 + j for j in range(3))
        self._Lambda2 = {1: self._Lambda2, -1: tuple(value**(-1) for value in self._Lambda2)}

    def __init__(self, gp: GPInterface, **kwargs: Any):
        """ Construct a ClosedIndex object.

        Args:
            gp: The gp to analyze.
            **kwargs: The calculation options to override OPTIONS.
        """
        super().__init__()
        self._gp = gp
        self._options = self.OPTIONS | kwargs
        # Unwrap parameters
        self._L, self._M, self._N = self._gp.L, self._gp.M, self._gp.N
        self._E = tf.constant(self._gp.likelihood.params.variance, dtype=FLOAT())
        self._F = tf.transpose(tf.constant(self._gp.kernel.params.variance, dtype=FLOAT()))     # To convert (1,L) to (L,1)
        self._lambda = tf.constant(self._gp.kernel.params.lengthscales, dtype=FLOAT())
        self._calc_Lambda2_()
        self._pmF = tf.sqrt((2 * np.pi)**(self._M) * tf.reduce_prod(self._Lambda2[1][0], axis=-1)) * self._F
        self._mu_phi_mu = {'pre-factor': tf.sqrt((2 * np.pi)**(self._M) *
                                                 tf.reduce_prod(self._Lambda2[1][0], axis=-1) * tf.reduce_prod(self._Lambda2[-1][2], axis=-1))}
        # Cache the training data kernel
        self._KNoisy_Cho = self._gp.KNoisy_Cho
        self._KNoisyInv_Y = self._gp.KNoisyInv_Y
        # Calculate selected constants
        self._g0 = self._pmF[..., tf.newaxis] * Gaussian.pdf(mean=self._gp.X[tf.newaxis, tf.newaxis, ...],
                                                                 variance_cho=tf.sqrt(self._Lambda2[1][1]), is_variance_diagonal=True, LBunch=2)
        self._KYg0 = self._g0 * tf.reshape(self._KNoisyInv_Y, [self._L, 1, self._N] if self._gp.kernel.is_independent else [1, self._L, self._N])
        self._KYg0_summed = tf.reduce_sum(self._KYg0, axis=[1, 2])
        # if self._options['is_T_calculated']:
        #     self._g0_2 = tf.einsum('LJ, lj -> LJlj', self._g0_dag, self._g0_dag)
        # Calculate the initial results
        self._Theta = tf.eye(self._M, dtype=FLOAT())
        self._m = self._M
        self._calculate()


class RotatedClosedIndex(ClosedIndex):
    """ Encapsulates the calculation of closed Sobol indices with a rotation U = Theta X."""

    def _matrix_inverse(self, tensor: TF.Tensor, I: tf.Tensor = None) -> TF.Tensor:
        """ Invert the inner matrix of an (L,L,M,M) or (L,L,L,L,M,M) Tensor.

        Args:
            tensor: A tensor whose shape matches identity.
            I: Supply the (L,L,M,M) identity matrix, otherwise the (L,L,L,L,M,M) identity matrix is used.
        Returns: The inner matrix inverse of tensor.
        """
        if I is None:
            I = tf.eye(self._M, batch_shape=[1, 1, 1, 1], dtype=FLOAT())
            ein = 'IiLlmM, IiLlmJ -> IiLlMJ'
        else:
            ein = 'LlmM, LlmJ -> LlMJ'
        result = tf.linalg.cholesky(tensor)
        result = tf.linalg.triangular_solve(result, I)
        return tf.einsum(ein , result, result)

    def rotate_and_calculate(self, Theta: TF.Matrix) -> Dict[str, TF.Tensor]:
        """ Rotate the input basis by Theta, calculate and return all quantities which do not depend on marginalization,
            but will need to be marginalized.

        Args:
            Theta: An (M,M) matrix to rotate the inputs to U = Theta X.

        Returns:

        """
        I = tf.eye(self._M, batch_shape=[1, 1], dtype=FLOAT())
        # First Moments
        G = tf.einsum('Mm, Llm, Nm -> LlNM', Theta, self._Lambda2[-1][1], self._gp.X)
        Phi = tf.einsum('Mm, Llm, Jm -> LlMJ', Theta, self._Lambda2[-1][1], Theta)
        Gamma = I - Phi
        # Second Moments
        Upsilon = tf.einsum('Mm, Llm, Llm, Jm -> LlMJ', Theta, self._Lambda2[1][1], self._Lambda2[-1][2], Theta)
        Gamma_inv = self._matrix_inverse(Gamma, I)
        print(tf.einsum('LlMm, LlmJ -> LlMJ', Gamma_inv, Gamma))
        Upsilon_inv = self._matrix_inverse(Upsilon, I)
        print(tf.einsum('LlMm, LlmJ -> LlMJ', Upsilon_inv, Upsilon))
        Pi = self._matrix_inverse(tf.einsum('LlMm, Llmj, LljJ -> LlMJ', Phi, Gamma_inv, Phi) + Upsilon_inv, I)

    def rotate_and_marginalize(self, Theta: TF.Matrix, G: TF.Tensor, Gamma: TF.Tensor, Upsilon: TF.Tensor):
        # Expected Value
        Sigma = tf.expand_dims(tf.expand_dims(Gamma, axis=2), axis=2) + Gamma[tf.newaxis, tf.newaxis, ...]
        Psi = Sigma - tf.einsum('IiMm, LlmJ -> IiLlMJ', Gamma, Gamma)
        SigmaPsi = tf.einsum('IiLlMm, IiLlmJ -> IiLlMJ', Sigma, Psi)
        Gamma_reshape = tf.expand_dims(Gamma, 2)
        SigmaG = tf.einsum('IinMm, LlNm -> IinLlNM', Gamma_reshape, G) + tf.einsum('IinMm, LlNm -> LlNIinM', Gamma_reshape, G)
        # Variance
        sqrt_1_Upsilon = tf.linalg.band_part(tf.linalg.cholesky(I - Upsilon), -1, 0)
