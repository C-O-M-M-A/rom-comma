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

import sym as sym

from romcomma.base.definitions import *
from romcomma.gpr.models import GPInterface
from abc import ABC


LOG2PI = tf.math.log(tf.constant(2 * np.pi, dtype=FLOAT()))
Triad = Dict[str, TF.Tensor]
TriadOfTriads = Dict[str, Triad]


class Gaussian(ABC):
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
        result = ((ordinate / variance_cho)[..., tf.newaxis] if is_variance_diagonal
                  else tf.linalg.triangular_solve(variance_cho, ordinate[..., tf.newaxis], lower=True))
        result = tf.einsum('...oz, ...oz -> ...', result, result)
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


class Dagger(ABC):
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
            self._Lambda2 = tf.expand_dims(tf.einsum('LM, LM -> LM', self._Lambda, self._Lambda), axis=1)
        else:
            self._Lambda2 = tf.einsum('LM, lM -> LlM', self._Lambda, self._Lambda)
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
        self._Lambda = tf.constant(self._gp.kernel.params.lengthscales, dtype=FLOAT())
        self._Lambda2_()
        self._pmF = tf.sqrt((2 * np.pi)**(self._M) * tf.reduce_prod(self._Lambda2[1][0], axis=-1)) * self._F
        # Cache the training data kernel
        self._KNoisy_Cho = self._gp.K_cho
        self._KNoisyInv_Y = self._gp.K_inv_Y
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
            is_T_calculated: If False, T and W return TF.NaN.
            is_T_diagonal: If True, only the S.shape-diagonal elements of T and W are calculated.
                In other words the variance of each element of S is calculated, but cross-covariances are not.
            is_T_partial: If True this effectively forces W[`m`][`M`] = W[`M`][`M`] = 0, as if the full ['M'] model is variance free.
        """
        return {'is_T_calculated': True, 'is_T_diagonal': False, 'is_T_partial': False}

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
        self._Gamma_reshape = tf.expand_dims(self._Gamma, axis=2)
        # FIXME: Debug
        print(f'_G = {self._G.shape}    _Phi = {self._Phi.shape}    _Gamma = {self._Gamma.shape}    _Gamma_reshape = {self._Gamma_reshape.shape}')
        print(f'_G sym {sym_check(self._G, [1,0,2,3])}    _Phi sym {sym_check(self._Phi, [1,0,2])}    _Gamma sym {sym_check(self._Gamma, [1,0,2])}')
        # print(f'_Phi = {self._Phi}')
        # Second Moments
        if self._options['is_T_calculated']:
            if self._options['is_T_diagonal']:
                self._Upsilon = self._Lambda2_diag[1][1] * self._Lambda2_diag[-1][2]
                self._Pi = 1 - self._Lambda2_diag[-1][1]
                self._mu_phi_mu['pre-factor'] = (tf.sqrt((2 * np.pi)**self._M * tf.reduce_prod(self._Lambda2_diag[1][0] * self._Lambda2_diag[-1][2], axis=-1))
                                                 * self._F)
            else:
                self._Upsilon = self._Lambda2[1][1] * self._Lambda2[-1][2]
                self._Pi = self._Gamma
                self._mu_phi_mu['pre-factor'] = (tf.sqrt((2 * np.pi)**self._M * tf.reduce_prod(self._Lambda2[1][0] * self._Lambda2[-1][2], axis=-1))
                                                 * self._F)
            # FIXME: Debug
            print(f'_Upsilon = {self._Upsilon.shape}    _Pi = {self._Pi.shape}')
            print(f'_Upsilon sym {sym_check(self._Upsilon, [1,0,2])}    _Pi sym {sym_check(self._Pi, [1,0,2])}')
            print(f'_mu_phi_mu["pre-factor"] = {self._mu_phi_mu["pre-factor"].shape}    _Gamma_reshape = {self._Gamma_reshape.shape}')
            print(f'_mu_phi_mu["pre-factor"] = {self._mu_phi_mu["pre-factor"]}')
        # Expected Value
        self._calculate_expectation()
        # Variance
        if self._options['is_T_calculated']:
            self._calculate_variance()

    def _calculate_expectation(self):
        self._Sigma = tf.expand_dims(self._Gamma_reshape, axis=2) + self._Gamma[tf.newaxis, tf.newaxis, ...]
        self._Psi = self._Sigma - tf.einsum('lLM, jJM -> lLjJM', self._Gamma, self._Gamma)
        self._SigmaPsi = tf.einsum('lLjJM, lLjJM -> lLjJM', self._Sigma, self._Psi)
        self._SigmaG = tf.einsum('jJnM, lLNM -> lLNjJnM', self._Gamma_reshape, self._G) + tf.einsum('lLNM, jJnM -> lLNjJnM', self._Gamma_reshape, self._G)
        Sigma_pdf, Sigma_det = Gaussian.pdf_with_det(mean=self._G, ordinate=self._G, variance_cho=tf.sqrt(self._Sigma), is_variance_diagonal=True, LBunch=2)
        SigmaPsi_pdf, SigmaPsi_det = Gaussian.pdf_with_det(mean=self._SigmaG, variance_cho=tf.sqrt(self._SigmaPsi), is_variance_diagonal=True, LBunch=2)
        self._H = (Sigma_pdf / SigmaPsi_pdf) * (Sigma_det / SigmaPsi_det)**2
        self._V['0'] = tf.einsum('l, j -> lj', self._KYg0_summed, self._KYg0_summed)
        self._V['M'] = tf.einsum('lLN, lLNjJn, jJn -> lj', self._KYg0, self._H, self._KYg0) - self._V['0']
        # FIXME: Debug
        print(f'_Sigma = {self._Sigma.shape}    _Psi = {self._Psi.shape}    _SigmaPsi = {self._SigmaPsi.shape}')
        print(f'_Sigma sym {sym_check(self._Sigma, [1, 0, 3, 2, 4])}    _Psi sym {sym_check(self._Psi, [1, 0, 3, 2, 4])}    _SigmaPsi sym {sym_check(self._SigmaPsi, [1, 0, 3, 2, 4])}')
        print(f'_Sigma sym {sym_check(self._Sigma, [2, 3, 0, 1, 4])}    _Psi sym {sym_check(self._Psi, [2, 3, 0, 1, 4])}    _SigmaPsi sym {sym_check(self._SigmaPsi, [2, 3, 0, 1, 4])}')
        print(f'_Sigma sym {sym_check(self._Sigma, [3, 2, 1, 0, 4])}    _Psi sym {sym_check(self._Psi, [3, 2, 1, 0, 4])}    _SigmaPsi sym {sym_check(self._SigmaPsi, [3, 2, 1, 0, 4])}')
        print(f'_Sigma sym {sym_check(self._Sigma, [3, 1, 2, 0, 4])}    _Psi sym {sym_check(self._Psi, [3, 1, 2, 0, 4])}    _SigmaPsi sym {sym_check(self._SigmaPsi, [3, 1, 2, 0, 4])}')
        print(f'_SigmaG = {self._SigmaG.shape}    Sigma_pdf = {Sigma_pdf.shape}    SigmaPsi_pdf = {SigmaPsi_pdf.shape}')
        if not self.gp.kernel.is_independent:
            print(f'_SigmaG sym {sym_check(self._SigmaG, [1, 0, 2, 4, 3, 5, 6])}  Sigma_pdf sym {sym_check(Sigma_pdf, [1, 0, 2, 4, 3, 5])}  SigmaPsi_pdf sym {sym_check(SigmaPsi_pdf, [1, 0, 2, 4, 3, 5])}')
            print(f'_SigmaG sym {sym_check(self._SigmaG, [3, 4, 5, 0, 1, 2, 6])}  Sigma_pdf sym {sym_check(Sigma_pdf, [3, 4, 5, 0, 1, 2])}  SigmaPsi_pdf sym {sym_check(SigmaPsi_pdf, [3, 4, 5, 0, 1, 2])}')
            print(f'_SigmaG sym {sym_check(self._SigmaG, [4, 3, 5, 1, 0, 2, 6])}  Sigma_pdf sym {sym_check(Sigma_pdf, [4, 3, 5, 1, 0, 2])}  SigmaPsi_pdf sym {sym_check(SigmaPsi_pdf, [4, 3, 5, 1, 0, 2])}')
            print(f'_SigmaG sym {sym_check(self._SigmaG, [4, 3, 2, 1, 0, 5, 6])}  Sigma_pdf sym {sym_check(Sigma_pdf, [4, 3, 2, 1, 0, 5])}  SigmaPsi_pdf sym {sym_check(SigmaPsi_pdf, [4, 3, 2, 1, 0, 5])}')
            print(f'Sigma_det sym {sym_check(Sigma_det, [1, 0, 2, 4, 3, 5])}  SigmaPsi_det sym {sym_check(SigmaPsi_det, [1, 0, 2, 4, 3, 5])}')
            print(f'Sigma_det sym {sym_check(Sigma_det, [3, 4, 5, 0, 1, 2])}  SigmaPsi_det sym {sym_check(SigmaPsi_det, [3, 4, 5, 0, 1, 2])}')
            print(f'Sigma_det sym {sym_check(Sigma_det, [4, 3, 5, 1, 0, 2])}  SigmaPsi_det sym {sym_check(SigmaPsi_det, [4, 3, 5, 1, 0, 2])}')
            print(f'Sigma_det sym {sym_check(Sigma_det, [4, 3, 2, 1, 0, 5])}  SigmaPsi_det sym {sym_check(SigmaPsi_det, [4, 3, 2, 1, 0, 5])}')
        print(f'Sigma_det = {Sigma_det.shape}    SigmaPsi_det = {SigmaPsi_det.shape}')
        # print(f'Sigma_det = {Sigma_det}')
        # print(f'SigmaPsi_det = {SigmaPsi_det}')
        print(f'_H = {self._H.shape}    _V["0"] = {self._V["0"].shape}    _V["M"] = {self._V["M"].shape}')
        if not self.gp.kernel.is_independent:
            print(f'self._H sym {sym_check(self._H, [1, 0, 2, 4, 3, 5])}')
            print(f'self._H sym {sym_check(self._H, [3, 4, 5, 0, 1, 2])}')
            print(f'self._H sym {sym_check(self._H, [4, 3, 5, 1, 0, 2])}')
            print(f'self._H sym {sym_check(self._H, [3, 1, 2, 0, 4, 5])}')
        print(f'_V["0"] = {self._V["0"]}')
        print(f'_V["M"] = {self._V["M"]}')

    def _calculate_variance(self):
        self._Gamma_reshape = tf.expand_dims(self._Gamma_reshape, axis=1)
        Phi_ein = 'iiM' if self._Phi.shape[0] == self._Phi.shape[1] else 'ijM'
        self._sqrt_1_Upsilon = tf.sqrt(self._Lambda2[-1][2])  # = sqrt(1 - Upsilon)
        if self._options['is_T_diagonal']:
            self._Omega = tf.expand_dims(tf.einsum(f'kJM, {Phi_ein} -> ikJM', self._Phi, self._Phi), axis=1)
        else:
            self._Omega = tf.einsum('kJM, ijM -> ijkJM', self._Phi, self._Phi)
        self._B = (tf.einsum('kJM, ijM, kJM -> ijkJM', self._Phi, self._Pi, self._Phi)
                   + tf.einsum('kJM, kJM -> kJM', self._Gamma, self._Phi)[tf.newaxis, tf.newaxis, ...])
        self._C = self._Gamma_reshape / (self._Gamma_reshape + tf.einsum('lLM, ijM -> liLjM', self._Phi, self._Upsilon))
        self._C = tf.einsum('ijM, liLjM -> liLjM', self._Upsilon, self._C)
        self._D = (tf.einsum('kKM, jJM, kKM -> jJkKM', self._Phi, self._Gamma, self._Phi)
                   + tf.expand_dims(tf.expand_dims(tf.einsum('kKM, kKM -> kKM', self._Gamma, self._Phi), axis=0), axis=0))
        g_factor = self._KYg0 / Gaussian.pdf(self._G, tf.sqrt(self._Phi), True)
        # FIXME: Debug
        print(f'_Gamma_reshape = {self._Gamma_reshape.shape}    g_factor = {g_factor.shape}    Phi_ein = {Phi_ein}')
        print(f'_sqrt_1_Upsilon = {self._sqrt_1_Upsilon.shape}    _Omega = {self._Omega.shape}')
        print(f'_B = {self._B.shape}    _C = {self._C.shape}    _D = {self._D.shape}')
        self._calculate_mu_phi_mu(g_factor)
        self._calculate_mu_psi_mu(g_factor)
        self._V2MM = {'MM': tf.einsum('li, jk -> lijk', self._V['M'], self._V['M'])}
        self._A = {key: self._mu_phi_mu[key] + self._mu_psi_mu[key] for key in self._keys['A']}
        if not self._options['is_T_partial']:
            self._WMM = self._A['MM'] - 2 * self._A['M0'] + self._A['00']
            # FIXME: Debug
            # print(f'_WMM = {self._WMM.shape}')
            # print(f'_WMM = {self._WMM}')
            # print(f'_A["MM"] = {self._A["MM"]}')
            # print(f'_A["M0"] = {self._A["M0"]}')
            # print(f'_A["00"] = {self._A["00"]}')
            print(f'_mu_psi_mu["00"] = {self._mu_psi_mu["00"]}')
            print(f'_mu_psi_mu["M0"] = {self._mu_psi_mu["M0"]}')
            print(f'_mu_psi_mu["MM"] = {self._mu_psi_mu["MM"]}')
            # print(f'_A["00"] sym {sym_check(self._A["00"], [1, 0, 3, 2])}    _A["M0"] sym {sym_check(self._A["M0"], [1, 0, 3, 2])}    _A["MM"] sym {sym_check(self._A["MM"], [1, 0, 3, 2])}')
            # print(f'_A["00"] sym {sym_check(self._A["00"], [2, 3, 0, 1])}    _A["M0"] sym {sym_check(self._A["M0"], [2, 3, 0, 1])}    _A["MM"] sym {sym_check(self._A["MM"], [2, 3, 0, 1])}')
            # print(f'_A["00"] sym {sym_check(self._A["00"], [3, 2, 1, 0])}    _A["M0"] sym {sym_check(self._A["M0"], [3, 2, 1, 0])}    _A["MM"] sym {sym_check(self._A["MM"], [3, 2, 1, 0])}')
            # print(f'_A["00"] sym {sym_check(self._A["00"], [3, 1, 2, 0])}    _A["M0"] sym {sym_check(self._A["M0"], [3, 1, 2, 0])}    _A["MM"] sym {sym_check(self._A["MM"], [3, 1, 2, 0])}')
            print(f'_mu_phi_mu["00"] sym {sym_check(self._mu_phi_mu["00"], [3, 1, 2, 0])}    _mu_phi_mu["M0"] sym {sym_check(self._mu_phi_mu["M0"], [3, 1, 2, 0])}    _mu_phi_mu["MM"] sym {sym_check(self._mu_phi_mu["MM"], [3, 1, 2, 0])}')
            print(f'_mu_phi_mu["00"] sym {sym_check(self._mu_phi_mu["00"], [0, 2, 1, 3])}    _mu_phi_mu["M0"] sym {sym_check(self._mu_phi_mu["M0"], [0, 2, 1, 3])}    _mu_phi_mu["MM"] sym {sym_check(self._mu_phi_mu["MM"], [0, 2, 1, 3])}')
            # print(f'_mu_phi_mu["00"] sym {sym_check(self._mu_phi_mu["00"], [3, 2, 1, 0])}    _mu_phi_mu["M0"] sym {sym_check(self._mu_phi_mu["M0"], [3, 2, 1, 0])}    _mu_phi_mu["MM"] sym {sym_check(self._mu_phi_mu["MM"], [3, 2, 1, 0])}')
            # print(f'_mu_phi_mu["00"] sym {sym_check(self._mu_phi_mu["00"], [3, 1, 2, 0])}    _mu_phi_mu["M0"] sym {sym_check(self._mu_phi_mu["M0"], [3, 1, 2, 0])}    _mu_phi_mu["MM"] sym {sym_check(self._mu_phi_mu["MM"], [3, 1, 2, 0])}')
            print(f'_mu_psi_mu["00"] sym {sym_check(self._mu_psi_mu["00"], [3, 1, 2, 0])}    _mu_psi_mu["M0"] sym {sym_check(self._mu_psi_mu["M0"], [3, 1, 2, 0])}    _mu_psi_mu["MM"] sym {sym_check(self._mu_psi_mu["MM"], [3, 1, 2, 0])}')
            print(f'_mu_psi_mu["00"] sym {sym_check(self._mu_psi_mu["00"], [0, 2, 1, 3])}    _mu_psi_mu["M0"] sym {sym_check(self._mu_psi_mu["M0"], [0, 2, 1, 3])}    _mu_psi_mu["MM"] sym {sym_check(self._mu_psi_mu["MM"], [0, 2, 1, 3])}')
            # print(f'_mu_psi_mu["00"] sym {sym_check(self._mu_psi_mu["00"], [3, 2, 1, 0])}    _mu_psi_mu["M0"] sym {sym_check(self._mu_psi_mu["M0"], [3, 2, 1, 0])}    _mu_psi_mu["MM"] sym {sym_check(self._mu_psi_mu["MM"], [3, 2, 1, 0])}')
            # print(f'_mu_psi_mu["00"] sym {sym_check(self._mu_psi_mu["00"], [3, 1, 2, 0])}    _mu_psi_mu["M0"] sym {sym_check(self._mu_psi_mu["M0"], [3, 1, 2, 0])}    _mu_psi_mu["MM"] sym {sym_check(self._mu_psi_mu["MM"], [3, 1, 2, 0])}')

    def _calculate_mu_phi_mu(self, g_factor: TF.Tensor):
        if self._options['is_T_diagonal']:
            self._mu_phi_mu['00'] = tf.einsum('ij, l, l -> lij', self._mu_phi_mu['pre-factor'], self._KYg0_summed, self._KYg0_summed)[..., tf.newaxis]
        else:
            self._mu_phi_mu['00'] = tf.einsum('ij, l, k -> lijk', self._mu_phi_mu['pre-factor'], self._KYg0_summed, self._KYg0_summed)
        if not self._options['is_T_partial']:
            sqrt_1_Upsilon_pdf = self._sqrt_1_Upsilon_pdf(True, self._sqrt_1_Upsilon, self._G, self._Phi)
            if self._options['is_T_diagonal']:
                self._mu_phi_mu['M0'] = tf.einsum('lLN, liLNj -> liLj', self._KYg0, sqrt_1_Upsilon_pdf)
                self._mu_phi_mu['MM'] = tf.einsum('lJn, liLNjkJn -> liLNjk', g_factor,
                                                  self._Omega_pdf(True, self._B, self._C, self._G, self._Omega, 1 / self._Gamma))
            else:
                self._mu_phi_mu['M0'] = tf.einsum('ij, lLN, liLNj, k -> lijk',
                                                  self._mu_phi_mu['pre-factor'], self._KYg0, sqrt_1_Upsilon_pdf, self._KYg0_summed)
                print(f'_mu_phi_mu["M0"] sym {sym_check(self._mu_phi_mu["M0"], [3, 1, 2, 0])}    _mu_phi_mu["M0"] sym {sym_check(self._mu_phi_mu["M0"], [3, 1, 2, 0])}    _mu_phi_mu["00"] sym {sym_check(self._mu_phi_mu["00"], [3, 1, 2, 0])}')
                self._mu_phi_mu['M0'] += tf.transpose(self._mu_phi_mu['M0'], [3, 0, 2, 1])
                self._mu_phi_mu['MM'] = tf.einsum('kJn, liLNjkJn -> liLNjk', g_factor,
                                                  self._Omega_pdf(True, self._B, self._C, self._G, self._Omega, 1 / self._Gamma))
            self._mu_phi_mu['MM'] = tf.einsum('lLN, liLNj, liLNjk -> lijk', self._KYg0, sqrt_1_Upsilon_pdf, self._mu_phi_mu['MM'])
            self._mu_phi_mu['MM'] += self._mu_phi_mu['00'] - self._mu_phi_mu['M0']
            if self._gp.kernel.is_independent:
                self._mu_phi_mu['MM'] = tf.transpose(tf.linalg.diag(tf.squeeze(tf.transpose(self._mu_phi_mu['MM'], [0, 3, 1, 2]), axis=-1)), [0, 2, 3, 1])
            self._mu_phi_mu['MM'] += (tf.transpose(self._mu_phi_mu['MM'], [1, 0, 2, 3]) + tf.transpose(self._mu_phi_mu['MM'], [0, 1, 3, 2])
                                      + tf.transpose(self._mu_phi_mu['MM'], [1, 0, 3, 2]))
            # FIXME: Debug
            print(f'sqrt_1_Upsilon_pdf = {sqrt_1_Upsilon_pdf.shape}       _mu_phi_mu["M0"] = {self._mu_phi_mu["M0"].shape}  _mu_phi_mu["MM"] = {self._mu_phi_mu["MM"].shape}')

    def _calculate_mu_psi_mu(self, g_factor: TF.Tensor):
        D_cho = tf.sqrt(self._D)
        factor = {}
        factor['0'] = tf.einsum('l, iIn -> liIn', self._KYg0_summed, self._g0)
        shape = factor['0'].shape[0:2].as_list() + [-1, 1]
        if not self._options['is_T_partial']:
            pdf = Gaussian.pdf(mean=self._G, variance_cho=D_cho, is_variance_diagonal=True, ordinate=self._G)
            factor['M'] = tf.einsum('lLN, lLNiIn -> liIn', g_factor, pdf)
        self._mu_psi_mu_factor = {key: tf.squeeze(tf.linalg.triangular_solve(self._K_cho, tf.reshape(factor[key], shape)), axis=-1)
                                  for key in self._keys['factor']}
        if self._options['is_T_diagonal']:
            self._mu_psi_mu = {key: tf.einsum('liIn, liIn -> li', factor[key[0]], factor[key[1]])[..., tf.newaxis, tf.newaxis]
                               for key in self._keys['A']}
        else:
            if self._gp.kernel.is_independent:
                self._mu_psi_mu = {key: tf.transpose(tf.linalg.diag(tf.transpose(tf.einsum('liIn, kiIn -> lik', factor[key[0]], factor[key[1]]),
                                                                                 [0, 2, 1])), [0, 2, 3, 1]) for key in self._keys['A']}
            else:
                self._mu_psi_mu = {key: tf.einsum('liIn, kjIn -> lijk', factor[key[0]], factor[key[1]])
                                   for key in self._keys['A']}
        # FIXME: Debug
        if not self._options['is_T_partial']:
            print(f'factor["0"] = {factor["0"]}')
            print(f'factor["M"] = {factor["M"]}')
            print(f'factor["0"] = {factor["0"].shape}       factor["M"] = {factor["M"].shape}')
            print(f'_mu_psi_mu_factor["0"] = {self._mu_psi_mu_factor["0"].shape}       _mu_psi_mu_factor["M"] = {self._mu_psi_mu_factor["M"].shape}')
            print(f'_mu_psi_mu["00"] = {self._mu_psi_mu["00"].shape}       _mu_psi_mu["M0"] = {self._mu_psi_mu["M0"].shape}  _mu_psi_mu["MM"] = {self._mu_psi_mu["MM"].shape}')

    def _Omega_pdf(self, is_diagonal: bool, B: TF.Tensor, C: TF.Tensor, G: TF.Tensor, Omega: TF.Tensor, Gamma_inv: TF.Tensor) -> TF.Tensor:
        # The assumption here is that m >= m'. This is important
        ein = 'ijkJM, liLjM, lLM, lLNM -> liLNjkJM' if is_diagonal else 'ijkJMa, liLjab, lLbc, lLNc -> liLNjkJM'
        mean = tf.einsum(ein, Omega[..., :C.shape[-1]], C, Gamma_inv, G)
        mean = tf.expand_dims(mean, axis=7) - G[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]
        ein = 'ijkJM, liLjM, ijkJM -> liLjkJM' if is_diagonal else 'ijkJMa, liLjab, ijkJmb -> liLjkJMm'
        variance = tf.einsum(ein, Omega[..., :C.shape[-1]], C, Omega[..., :C.shape[-1]]) + tf.expand_dims(tf.expand_dims(B, axis=1), axis=0)
        variance = tf.sqrt(variance) if is_diagonal else tf.linalg.cholesky(variance)
        return Gaussian.pdf(mean, variance, is_diagonal, LBunch=3)

    def _sqrt_1_Upsilon_pdf(self, is_diagonal: bool, sqrt_1_Upsilon: TF.Tensor, G: TF.Tensor, Phi: TF.Tensor) -> TF.Tensor:
        ein = 'ijM, lLNM -> liLNjM' if is_diagonal else 'ijMm, lLNm -> liLNjM'
        mean = tf.einsum(ein, sqrt_1_Upsilon, G)
        ein = 'ijM, lLM, ijM -> liLjM' if is_diagonal else 'ijMm, lLma, ijAa -> liLjMA'
        variance = 1 - tf.einsum(ein, sqrt_1_Upsilon, Phi, sqrt_1_Upsilon)
        variance = tf.sqrt(variance) if is_diagonal else tf.linalg.cholesky(variance)
        return Gaussian.pdf(mean, variance, is_diagonal, LBunch=3)
        # return tf.transpose(tf.linalg.diag(tf.transpose(tf.squeeze(result, axis=-1),
        #                                                 [0, 2, 3, 1])), [0, 3, 1, 2, 4]) if self.gp.kernel.is_independent else result

    def _calc_Lambda2_(self, is_diagonal: bool) -> dict[int, Tuple[TF.Tensor]]:
        if is_diagonal:
            result = tf.expand_dims(tf.einsum('LM, LM -> LM', self._Lambda, self._Lambda), axis=1)
        else:
            result = tf.einsum('LM, lM -> LlM', self._Lambda, self._Lambda)
        result = tuple(result + j for j in range(3))
        return {1: result, -1: tuple(value**(-1) for value in result)}

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
        self._F = tf.transpose(tf.constant(self._gp.kernel.params.variance, dtype=FLOAT()))     # To convert (1,L) to (L,1)
        self._Lambda = tf.constant(self._gp.kernel.params.lengthscales, dtype=FLOAT())
        self._Lambda2 = self._calc_Lambda2_(is_diagonal=self._gp.kernel.is_independent)
        self._Lambda2_diag = self._calc_Lambda2_(is_diagonal=True)
        # Cache the training data kernel
        self._K_cho = self._gp.K_cho
        self._K_inv_Y = self._gp.K_inv_Y
        # Calculate selected constants
        self._g0 = Gaussian.pdf(mean=self._gp.X[tf.newaxis, tf.newaxis, ...],
                                                             variance_cho=tf.sqrt(self._Lambda2[1][1]), is_variance_diagonal=True, LBunch=2)
        self._pmF = tf.sqrt((2 * np.pi) ** (self._M) * tf.reduce_prod(self._Lambda2[1][0], axis=-1)) * self._F
        self._g0 *= self._pmF[..., tf.newaxis]
        self._KYg0 = self._g0 * self._K_inv_Y
        self._KYg0_summed = tf.einsum('lLN -> l', self._KYg0)
        # FIXME: Debug
        print()
        print(f'kernel.is_independent = {self._gp.kernel.is_independent}    is_T_diagonal = {self._options["is_T_diagonal"]}    is_T_partial = {self._options["is_T_partial"]}')
        # print(f'_Lambda2 = {self._Lambda2[1][0]}')
        # print(f'_Lambda2+1 = {self._Lambda2[1][1]}')
        print(f'_pmF = {self._pmF}') #    _Lambda2 = {self._Lambda2[1][1].shape}    _Lambda2_diag = {self._Lambda2_diag[1][1].shape}')
        # print(f'_g0_det sym {sym_check(self._g0_det, [1,0])}    _Lambda2 sym {sym_check(self._Lambda2[1][1], [1,0,2])}')
        # print(f'_K_cho = {self._K_cho.shape}    _K_inv_Y = {self._K_inv_Y.shape}')
        print(f'_KYg0_summed = {self._KYg0_summed}')
        print(f'_g0 = {self._g0.shape}  _KYg0 = {self._KYg0.shape}    _KYg0_summed = {self._KYg0_summed.shape}')
        # print(f'_g0 sym {sym_check(self._g0, [1,0,2])}  _KYg0 sym {sym_check(self._KYg0, [1,0,2])}')
        self._V = {}
        if self._options['is_T_calculated']:
            self._keys = {}
            if self._options['is_T_partial']:
                self._keys['factor'] = {'0'}
                self._keys['A'] = {'00'}
            else:
                self._keys['factor'] = {'0', 'M'}
                self._keys['A'] = {'00', 'M0', 'MM'}
            self._mu_phi_mu = {}
            self._mu_psi_mu = {}
            self._mu_psi_mu_factor = {}
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


def sym_check(tensor: TF.Tensor, transposition: List[int]) -> TF.Tensor:
    return tf.reduce_sum((tensor - tf.transpose(tensor, transposition))**2)
