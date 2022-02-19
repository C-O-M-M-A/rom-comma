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
#  PROCUREMENT OF SUBSTITUTE G00DS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Contains the calculation of a single coefficient of determination (closed Sobol index) without storing it."""

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.gpr.models import GPInterface
from abc import ABC


class Gaussian(ABC):
    """ Encapsulates the calculation of a Gaussian pdf. Not instantiatable."""

    TWO_PI = tf.constant(2 * np.pi, dtype=FLOAT())
    LOG_TWO_PI = tf.math.log(TWO_PI)

    @classmethod
    def pdf(cls, exponent: TF.Tensor, variance_cho_diagonal: TF.Tensor):
        """ Calculate the Gaussian pdf from the output of Gaussian.log_pdf.
        Args:
            exponent: The exponent in the Gaussian pdf.
            variance_cho_diagonal: The diagonal of the variance Cholesky decomposition.

        Returns: The Gaussian pdf.
        """
        return tf.exp(exponent) / tf.reduce_prod(variance_cho_diagonal, axis=-1)

    @classmethod
    def log_pdf(cls, mean: TF.Tensor, variance_cho: TF.Tensor, is_variance_diagonal: bool,
                ordinate: TF.Tensor = tf.constant(0, dtype=FLOAT()), LBunch: int = 2) -> Tuple[TF.Tensor, TF.Tensor]:
        """ Computes the logarithm of the un-normalized gaussian probability density, and the broadcast diagonal of variance_cho.
        Taking the product (tf.reduce_prod(variance_cho_diagonal, axis=-1) gives the normalization factor for the gaussian pdf.
        Batch dimensions of ordinate, mean and variance are internally broadcast to match each other.
        This function is used to minimize exponentiation, for efficiency and accuracy purposes, in calculating ratios of gaussian pdfs.

        Args:
            mean: Gaussian population mean. Should be of adequate rank to broadcast Ls.
            variance_cho: The lower triangular Cholesky decomposition of the Gaussian population variance. Is automatically broadcast to embrace Ns
            is_variance_diagonal: True if variance is an M-vector
            ordinate: The ordinate (z-value) to calculate the Gaussian density for. Should be of adequate rank to broadcast Ls. If not supplied, 0 is assumed.
            LBunch: The number of consecutive output (L) dimensions to count before inserting an N for broadcasting. Usually 2, sometimes 3.
        Returns: The tensor Gaussian pdf, and the diagonal of variance_cho.
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
        if is_variance_diagonal:
            exponent = (ordinate / variance_cho)
        else:
            exponent = tf.squeeze(tf.linalg.triangular_solve(variance_cho, ordinate[..., tf.newaxis], lower=True), axis=-1)
            variance_cho = tf.linalg.diag_part(variance_cho)
        exponent = - 0.5 * (cls.LOG_TWO_PI * ordinate.shape[-1] + tf.einsum('...o, ...o -> ...', exponent, exponent))
        return exponent, variance_cho


class ClosedIndex(gf.Module):
    """ Calculates closed Sobol Indices."""

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        """ Default calculation options. ``is_T_partial`` forces W[`m`][`M`] = W[`M`][`M`] = 0.

        Returns:
            is_T_calculated: If False, T and W are not calculated or returned.
            is_T_diagonal: If True, only the S.shape-diagonal elements of T and W are calculated.
                In other words the variance of each element of S is calculated, but cross-covariances are not.
            is_T_partial: If True this effectively asserts the full ['M'] model is variance free, so WmM is not calculated or returned.
        """
        return {'is_T_calculated': True, 'is_T_diagonal': False, 'is_T_partial': False}

    def _calculate(self):
        pre_factor = tf.sqrt((Gaussian.TWO_PI) ** (self.M) * tf.reduce_prod(self.Lambda2[1][0] / self.Lambda2[1][1], axis=-1)) * self.F
        self.g0, _ = Gaussian.log_pdf(mean=self.gp.X[tf.newaxis, tf.newaxis, ...],
                                       variance_cho=tf.sqrt(self.Lambda2[1][1]), is_variance_diagonal=True, LBunch=2)
        self.g0 = pre_factor[..., tf.newaxis] * tf.exp(self.g0)
        self.KYg0 = self.g0 * self.K_inv_Y
        self.KYg0_sum = tf.einsum('lLN -> l', self.KYg0)
        self.V = {'0': tf.einsum('l, j -> lj', self.KYg0_sum, self.KYg0_sum)}
        self.G = tf.einsum('lLM, NM -> lLNM', self.Lambda2[-1][1], self.gp.X)
        self.Phi = self.Lambda2[-1][1]
        self.Gamma = 1 - self.Phi
        # FIXME: Debug
        print()
        print(f'kernel.is_independent = {self.gp.kernel.is_independent}    is_T_diagonal = {self.options["is_T_diagonal"]}    is_T_partial = {self.options["is_T_partial"]}')
        self.dtype = set(['calculate']) | set([self.Lambda.dtype, self.Lambda2[1][1].dtype, self.Lambda2_diag[-1][1].dtype, self.K_cho.dtype, self.g0.dtype, self.KYg0.dtype])
        print(self.dtype)
        self.V['M'] = self._V(self.G, self.Gamma)
        if self.options['is_T_calculated']:
            self.Phi_ein = 'iiM' if self.Phi.shape[0] == self.Phi.shape[1] else 'ijM'
            if self.options['is_T_diagonal']:
                self.Upsilon = self.Lambda2_diag[1][1] * self.Lambda2_diag[-1][2]
                self.V2MM = tf.einsum('li, li -> li', self.V['M'], self.V['M'])[..., tf.newaxis, tf.newaxis]
                self.mu_phi_mu = {'pre-factor': tf.sqrt((Gaussian.TWO_PI) ** self.M *
                                                        tf.reduce_prod(self.Lambda2_diag[1][0] * self.Lambda2_diag[-1][2], axis=-1)) * self.F}
            else:
                self.Upsilon = self.Lambda2[1][1] * self.Lambda2[-1][2]
                self.V2MM = tf.einsum('li, jk -> lijk', self.V['M'], self.V['M'])
                self.mu_phi_mu = {'pre-factor': tf.sqrt((Gaussian.TWO_PI) ** self.M *
                                                        tf.reduce_prod(self.Lambda2[1][0] * self.Lambda2[-1][2], axis=-1)) * self.F}
                if self.gp.kernel.is_independent:
                    self.mu_phi_mu['pre-factor'] = tf.linalg.diag(tf.squeeze(self.mu_phi_mu['pre-factor'], axis=-1))
            self.mu_phi_mu['pre-factor'] = self.mu_phi_mu['pre-factor'][tf.newaxis, ..., tf.newaxis]
            self.Upsilon_log_pdf = self._Upsilon_log_pdf(self.G, self.Phi, self.Upsilon)
            self.G_log_pdf = Gaussian.log_pdf(mean=self.G, variance_cho=tf.sqrt(self.Phi), is_variance_diagonal=True, LBunch=2)
            self.Omega_log_pdf = self._Omega_log_pdf(self.Ms, self.Ms, self.G, self.Phi, self.Gamma, self.Upsilon)
            factor = tf.einsum('l, iIn -> liIn', self.KYg0_sum, self.g0)
            self.psi_factor = {'shape': factor.shape[:2].as_list() + [-1, 1]}
            self.psi_factor['0'] = tf.squeeze(tf.linalg.triangular_solve(self.K_cho, tf.reshape(factor, self.psi_factor['shape'])), axis=-1)
            self.psi_factor['M'] = self._psi_factor(self.G, self.Phi, self.G_log_pdf)
            # FIXME: Debug
            print(f'|psi_factor["0"]| = {magnitude(self.psi_factor["0"], "liS, liS")}')
            print(f'|psi_factor["M"]| = {magnitude(self.psi_factor["M"], "liS, liS")}')
            A_mislabelled = self._A(self._mu_phi_mu(self.G_log_pdf, self.Upsilon_log_pdf, self.Omega_log_pdf, self.Omega_log_pdf, is_constructor=True),
                                    self._mu_psi_mu(self.psi_factor['M'], is_constructor=True))
            self.A = {'00': A_mislabelled.pop('Mm')}
            if not self.options['is_T_partial']:
                self.W = self._W(**A_mislabelled)
                self.A = self.A | A_mislabelled
                # FIXME: Debug
                # print(f'|self.A["mm"]| = {magnitude(self.A["mm"])}')
                # print(f'|self.A["m0"]| = {magnitude(self.A["m0"])}')
                # print(f'|self.A["00"]| = {magnitude(self.A["00"])}')
                # print(f'|self.W["mm"]| = {magnitude(self.W["mm"])}')

    def marginalize(self, m: TF.Slice) -> Dict[str, Dict[str: TF.Tensor]]:
        """ Calculate everything.
        Args:
            m: A Tf.Tensor pair of ints indicating the slice [m[0]:m[1]].
        Returns:

        """
        G, Phi, Gamma = self.G, self.Phi, self.Gamma
        result = {'V': self._V(G[..., m[0]:m[1]], Gamma[..., m[0]:m[1]])}
        result['S'] = result['V'] / self.V['M']
        if self.options['is_T_calculated']:
            Upsilon = self.Upsilon
            G_m = G[..., m[0]:m[1]]
            Phi_mm = Phi[..., m[0]:m[1]]
            Upsilon_log_pdf = self._Upsilon_log_pdf(G_m, Phi_mm, Upsilon[..., m[0]:m[1]])
            G_log_pdf = Gaussian.log_pdf(G_m, tf.sqrt(Phi_mm), is_variance_diagonal=True, LBunch=2)
            Omega_log_pdf_M = self._Omega_log_pdf(self.Ms, m, G, Phi, Gamma, Upsilon)
            Omega_log_pdf_m = self._Omega_log_pdf(m, m, G, Phi, Gamma, Upsilon)
            psi_factor = self._psi_factor(G[..., m[0]:m[1]], Phi[..., m[0]:m[1]], G_log_pdf)
            result = result | self._T(result['V'],
                                      **self._W(**self._A(self._mu_phi_mu(G_log_pdf, Upsilon_log_pdf, Omega_log_pdf_M, Omega_log_pdf_m),
                                                          self._mu_psi_mu(psi_factor))))
        return result

    def _V(self, G: TF.Tensor, Gamma: TF.Tensor) -> TF.Tensor:
        Gamma_reshape = tf.expand_dims(Gamma, axis=2)
        Sigma = tf.expand_dims(Gamma_reshape, axis=2) + Gamma[tf.newaxis, tf.newaxis, ...]
        Psi = Sigma - tf.einsum('lLM, jJM -> lLjJM', Gamma, Gamma)
        SigmaPsi = tf.einsum('lLjJM, lLjJM -> lLjJM', Sigma, Psi)
        SigmaG = tf.einsum('jJnM, lLNM -> lLNjJnM', Gamma_reshape, G) + tf.einsum('lLNM, jJnM -> lLNjJnM', Gamma_reshape, G)
        Sigma_pdf, Sigma_diag = Gaussian.log_pdf(mean=G, ordinate=G, variance_cho=tf.sqrt(Sigma), is_variance_diagonal=True, LBunch=2)
        SigmaPsi_pdf, SigmaPsi_diag = Gaussian.log_pdf(mean=SigmaG, variance_cho=tf.sqrt(SigmaPsi), is_variance_diagonal=True, LBunch=2)
        H = Gaussian.pdf(Sigma_pdf - SigmaPsi_pdf, SigmaPsi_diag/Sigma_diag)
        V = tf.einsum('lLN, lLNjJn, jJn -> lj', self.KYg0, H, self.KYg0) - self.V['0']
        # FIXME: Debug
        # print(f'V = {V}')
        self.dtype = set(['V']) | self.dtype | set([G.dtype, Gamma.dtype, Sigma.dtype, Psi.dtype, SigmaPsi.dtype, SigmaG.dtype, Sigma_pdf.dtype, Sigma_diag.dtype, SigmaPsi_pdf.dtype, SigmaPsi_diag.dtype, H.dtype, V.dtype])
        print(self.dtype)
        return V

    def _T(self, Vm: TF.Tensor, mm: TF.Tensor, Mm: TF.Tensor = TF.NaN) -> Dict[str, TF.Tensor]:
        T = mm
        if not self.options['is_T_partial']:
            V_ratio = Vm / self.V['M']
            if self.options['is_T_diagonal']:
                V_ratio_2 = tf.einsum('li, li -> li', V_ratio, V_ratio)[..., tf.newaxis, tf.newaxis]
                V_ratio = V_ratio[..., tf.newaxis, tf.newaxis]
            else:
                V_ratio_2 = tf.einsum('li, jk -> lijk', V_ratio, V_ratio)
                V_ratio = tf.einsum('li, jk -> lijk', V_ratio, tf.ones_like(V_ratio))
            T += self.W['mm'] * V_ratio_2 - 2 * Mm * V_ratio
        return {'T': T / self.V2MM, 'Wmm': mm} if self.options['is_T_partial'] else {'T': T / self.V2MM, 'Wmm': mm, 'WmM': Mm}

    def _W(self, Om: TF.Tensor, m0: TF.Tensor, mm: TF.Tensor, Mm: TF.Tensor = TF.NOT_CALCULATED) -> Dict[str, TF.Tensor]:
        if self.options['is_T_diagonal']:
            W = {'mm': mm - 2 * m0 + self.A['00']}
        else:
            W = {'mm': mm - m0 - Om + self.A['00']}
        if not self.options['is_T_partial'] and Mm.dtype.is_floating:
            if self.options['is_T_diagonal']:
                W['Mm'] = Mm - self.A['m0'] - m0 + self.A['00']
            else:
                W['Mm'] = Mm - self.A['m0'] - Om + self.A['00']
        return W

    def _A(self, mu_phi_mu: TF.Tensor, mu_psi_mu: TF.Tensor) -> Dict[str, TF.Tensor]:
        A = mu_phi_mu - mu_psi_mu
        A = tf.concat([tf.transpose(A[0:1, ...], [0, 4, 3, 2, 1]), A], axis=0)  # Create Om from m0 symmetry of mu_phi_mu and mu_psi_mu.
        if self.options['is_T_diagonal']:
            A += tf.transpose(A, [0, 2, 1, 3, 4])
            A *= 2
        else:
            A += tf.transpose(A, [0, 2, 1, 3, 4]) + tf.transpose(A, [0, 1, 2, 4, 3]) + tf.transpose(A, [0, 2, 1, 4, 3])
        return {'Om': A[0], 'm0': A[1], 'mm': A[2], 'Mm': A[-1]} if A.shape[0] > 1 else {'Mm': A[0]}

    def _mu_psi_mu(self, psi_factor: TF.Tensor, is_constructor: bool = False) -> TF.Tensor:
        mu_psi_mu = []
        if self.options['is_T_diagonal']:
            if not is_constructor or (is_constructor and not self.options['is_T_partial']):
                mu_psi_mu += [tf.einsum('liS, liS -> li', psi_factor, self.psi_factor['0'])[..., tf.newaxis, tf.newaxis],
                             tf.einsum('liS, liS -> li', psi_factor, psi_factor)[..., tf.newaxis, tf.newaxis]]
            if is_constructor:
                mu_psi_mu += [tf.einsum('liS, liS -> li', self.psi_factor['0'], self.psi_factor['0'])[..., tf.newaxis, tf.newaxis]]
            else:
                if not self.options['is_T_partial']:
                    mu_psi_mu += [tf.einsum('liS, liS -> li', self.psi_factor['M'], psi_factor)[..., tf.newaxis, tf.newaxis]]
        else:
            if not is_constructor or (is_constructor and not self.options['is_T_partial']):
                mu_psi_mu += [tf.einsum('liS, kjS -> lijk', psi_factor, self.psi_factor['0']),
                             tf.einsum('liS, kjS -> lijk', psi_factor, psi_factor)]
            if is_constructor:
                mu_psi_mu += [tf.einsum('liS, kjS -> lijk', self.psi_factor['0'], self.psi_factor['0'])]
            else:
                if not self.options['is_T_partial']:
                    mu_psi_mu += [tf.einsum('liS, kjS -> lijk', self.psi_factor['M'], psi_factor)]
        return tf.stack(mu_psi_mu)

    def _psi_factor(self, G: TF.Tensor, Phi: TF.Tensor, G_log_pdf: Tuple[TF.Tensor, TF.Tensor]) -> TF.Tensor:
        D = Phi[tf.newaxis, tf.newaxis, ...] - tf.einsum('kKM, jJM, kKM -> kKjJM', Phi, Phi, Phi)
        log_pdf = list(Gaussian.log_pdf(mean=G, variance_cho=tf.sqrt(D), is_variance_diagonal=True, ordinate=G, LBunch=2))
        log_pdf[0] -= G_log_pdf[0][tf.newaxis, tf.newaxis, tf.newaxis, ...]
        log_pdf[1] /= G_log_pdf[1][tf.newaxis, tf.newaxis, tf.newaxis, ...]
        # FIXME: Debug
        # print(f'|g0| = {magnitude(self.g0, "kjS, kjS")}')
        # print(f'|KYg0| = {magnitude(self.KYg0, "kjS, kjS")}')
        # print(f'|K_inv_Y| = {magnitude(self.K_inv_Y, "kjS, kjS")}')
        # print(f'|Gaussian.pdf(*tuple(log_pdf))| = {magnitude(Gaussian.pdf(*tuple(log_pdf)), "kKnjJN, kKnjJN")}')
        # print(f'|det ratio| = {tf.sqrt(tf.reduce_prod(log_pdf[1], axis=-1))}')
        # print(f'|exponent| = {magnitude(log_pdf[0], "kKnjJN, kKnjJN")}')
        # print(f'|Gexp| = {magnitude(G_log_pdf[0], "kjS, kjS")}')
        # print(f'|G| = {magnitude(G)}')
        # print(f'|Gaussian.pdf(G)| = {magnitude(Gaussian.pdf(*tuple(G_log_pdf)) * tf.reduce_prod(G_log_pdf[1], axis=-1), "kjS, kjS")}')
        factor = tf.einsum('kKn, jJN, kKnjJN -> kjJN', self.KYg0, self.g0, Gaussian.pdf(*tuple(log_pdf)))
        # FIXME: Debug
        # print(f'before solve |factor| = {magnitude(factor)}')
        factor = tf.squeeze(tf.linalg.triangular_solve(self.K_cho, tf.reshape(factor, self.psi_factor['shape'])), axis=-1)
        # FIXME: Debug
        # print(f'after solve |factor| = {magnitude(factor, "kjS, kjS")}')
        return factor

    def _Upsilon_log_pdf(self, G: TF.Tensor, Phi: TF.Tensor, Upsilon: TF.Tensor) -> Tuple[TF.Tensor, TF.Tensor]:
        sqrt_1_Upsilon = tf.sqrt(1 - Upsilon)
        if self.options['is_T_diagonal']:
            mean = tf.expand_dims(tf.einsum(f'{self.Phi_ein}, lLNM -> liLNM', sqrt_1_Upsilon, G), axis=-2)
        else:
            mean = tf.einsum('ijM, lLNM -> liLNjM', sqrt_1_Upsilon, G)
        variance = 1 - tf.einsum('ijM, lLM, ijM -> liLjM', sqrt_1_Upsilon, Phi, sqrt_1_Upsilon)
        return Gaussian.log_pdf(mean, tf.sqrt(variance), is_variance_diagonal=True, LBunch=3)

    def _mu_phi_mu(self, G_log_pdf: TF.Tensor, Upsilon_log_pdf: TF.Tensor, Omega_log_pdf_M: TF.Tensor, Omega_log_pdf_m: TF.Tensor,
                   is_constructor: bool = False) -> TF.Tensor:
        mu_phi_mu = []
        if self.options['is_T_diagonal']:
            if not is_constructor or (is_constructor and not self.options['is_T_partial']):
                Omega_log_pdf_m[0] += Upsilon_log_pdf[0][..., tf.newaxis, tf.newaxis, tf.newaxis] - G_log_pdf[0]
                Omega_log_pdf_m[1] *= Upsilon_log_pdf[1][..., tf.newaxis, tf.newaxis, tf.newaxis] / G_log_pdf[1]
                mu_phi_mu += [tf.einsum('lLN, liLNj, l -> li', self.KYg0, Gaussian.pdf(*Upsilon_log_pdf), self.KYg0_sum)[..., tf.newaxis, tf.newaxis],
                              tf.einsum('lLN, liLNjkJn, kJn -> lijk', self.KYg0, Gaussian.pdf(*Omega_log_pdf_m), self.KYg0)]
            if is_constructor:
                mu_phi_mu += [tf.expand_dims(tf.expand_dims(tf.einsum('l, l -> l', self.KYg0_sum, self.KYg0_sum)[..., tf.newaxis], axis=1), axis=1)]
            else:
                if not self.options['is_T_partial']:
                    Omega_log_pdf_M[0] += self.Upsilon_log_pdf[0][..., tf.newaxis, tf.newaxis, tf.newaxis] - G_log_pdf[0]
                    Omega_log_pdf_M[1] *= self.Upsilon_log_pdf[1][..., tf.newaxis, tf.newaxis, tf.newaxis] / G_log_pdf[1]
                    mu_phi_mu += [tf.einsum('lLN, liLNjkJn, lJn -> lijk', self.KYg0, Gaussian.pdf(*Omega_log_pdf_M), self.KYg0)]
        else:
            if not is_constructor or (is_constructor and not self.options['is_T_partial']):
                Omega_log_pdf_m = list(Omega_log_pdf_m)
                Omega_log_pdf_m[0] += Upsilon_log_pdf[0][..., tf.newaxis, tf.newaxis, tf.newaxis] - G_log_pdf[0]
                Omega_log_pdf_m[1] *= Upsilon_log_pdf[1][..., tf.newaxis, tf.newaxis, tf.newaxis, :] / G_log_pdf[1]
                mu_phi_mu += [tf.einsum('lLN, liLNj, k -> lijk', self.KYg0, Gaussian.pdf(*Upsilon_log_pdf), self.KYg0_sum),
                              tf.einsum('lLN, liLNjkJn, kJn -> lijk', self.KYg0, Gaussian.pdf(*tuple(Omega_log_pdf_m)), self.KYg0)]
            if is_constructor:
                mu_phi_mu += [tf.expand_dims(tf.expand_dims(tf.einsum('l, k -> lk', self.KYg0_sum, self.KYg0_sum), axis=1), axis=1)]
            else:
                if not self.options['is_T_partial']:
                    Omega_log_pdf_M = list(Omega_log_pdf_M)
                    Omega_log_pdf_M[0] += self.Upsilon_log_pdf[0][..., tf.newaxis, tf.newaxis, tf.newaxis] - G_log_pdf[0]
                    Omega_log_pdf_M[1] *= self.Upsilon_log_pdf[1][..., tf.newaxis, tf.newaxis, tf.newaxis, :] / G_log_pdf[1]
                    mu_phi_mu += [tf.einsum('lLN, liLNjkJn, kJn -> lijk', self.KYg0, Gaussian.pdf(*tuple(Omega_log_pdf_M)), self.KYg0)]
        mu_phi_mu = [item * self.mu_phi_mu['pre-factor'] for item in mu_phi_mu]
        # FIXME: Debug
        for i, item in enumerate(mu_phi_mu):
            print(f'|mu_phi_mu[{i}]| = {magnitude(item)}')
        return tf.stack(mu_phi_mu)

    def _Omega_log_pdf(self, m: TF.Slice, mp: TF.Slice, G: TF.Tensor, Phi: TF.Tensor, Gamma: TF.Tensor,
                       Upsilon: TF.Tensor) -> Tuple[TF.Tensor, TF.Tensor]:
        """ The Omega integral for m=mp or m=[:M]. Does not apply when m=[0:0].

        Args:
            m: The marginalization m.
            mp: The marginalization m_primed.
            G: Un-marginalized.
            Phi: Un-marginalized.
            Gamma: Un-marginalized.
            Upsilon: Un-marginalized.
        Returns: Omega_log_pdf
        """
        Gamma_inv = 1 / Gamma
        Pi = 1 + Phi + tf.einsum('ijM, ijM, ijM -> ijM', Phi, Gamma_inv, Phi)
        Pi = 1 / Pi
        B = tf.einsum('kJM, kJM -> kJM', 1 - Phi, Phi)[tf.newaxis, tf.newaxis, ...]
        if self.options['is_T_diagonal']:
            B += tf.newaxis(tf.einsum(f'kJM, {self.Phi_ein}, kJM -> ikJM', Phi, Pi, Phi), axis=1)
            Gamma_reshape = tf.expand_dims(Gamma, axis=1)
            C = Gamma_reshape / (Gamma_reshape + tf.einsum(f'lLM, {self.Phi_ein} -> liLM', Phi, Upsilon))
            C = tf.expand_dims(tf.einsum(f'{self.Phi_ein}, liLM -> liLM', Upsilon, C), axis=3)
            Omega = tf.expand_dims(tf.einsum(f'{self.Phi_ein}, {self.Phi_ein}, {self.Phi_ein} -> iM', Pi, Phi, Gamma_inv), axis=1)
        else:
            B += tf.einsum('kJM, ijM, kJM -> ijkJM', Phi, Pi, Phi)
            Gamma_reshape = tf.expand_dims(tf.expand_dims(Gamma, axis=1), axis=3)
            C = Gamma_reshape / (Gamma_reshape + tf.einsum('lLM, ijM -> liLjM', Phi, Upsilon))
            C = tf.einsum('ijM, liLjM -> liLjM', Upsilon, C)
            Omega = tf.einsum('ijM, ijM, ijM -> ijM', Pi, Phi, Gamma_inv)
        Omega = tf.einsum('kJp, ijp -> ijkJp', Phi, Omega)
        mean = tf.einsum('ijkJM, liLjM, lLM, lLNM -> liLNjkJM', Omega, C, Gamma_inv, G)
        if m[1] < mp[1]:
            pad = [m[0], self.M - m[1]]
            mean = tf.pad(mean[..., m[0]:m[1]], pad)
            variance = (tf.expand_dims(tf.expand_dims(B, axis=0), axis=2) +
                        tf.pad(tf.einsum('ijkJM, liLjM, ijkJM -> liLjkJM', Omega, C, Omega)[..., m[0]:m[1]], pad))
        else:
            variance = tf.expand_dims(tf.expand_dims(B, axis=0), axis=2) + tf.einsum('ijkJM, liLjM, ijkJM -> liLjkJM', Omega, C, Omega)
        if mp is not self.Ms:
            variance = variance[..., mp[0]:mp[1]]
            mean = mean[..., mp[0]:mp[1]]
        mean = tf.expand_dims(mean, axis=7) - G[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]
        return Gaussian.log_pdf(mean=mean, variance_cho=tf.sqrt(variance), is_variance_diagonal=True, LBunch=3)

    def _Lambda2(self, is_diagonal: bool) -> dict[int, Tuple[TF.Tensor]]:
        if is_diagonal:
            result = tf.expand_dims(tf.einsum('LM, LM -> LM', self.Lambda, self.Lambda), axis=1)
        else:
            result = tf.einsum('LM, lM -> LlM', self.Lambda, self.Lambda)
        result = tuple(result + j for j in range(3))
        return {1: result, -1: tuple(value**(-1) for value in result)}

    def __init__(self, gp: GPInterface, **kwargs: Any):
        """ Construct a ClosedIndex object.

        Args:
            gp: The gp to analyze.
            **kwargs: The calculation options to override OPTIONS.
        """
        super().__init__()
        self.gp = gp
        self.options = self.OPTIONS | kwargs
        # Unwrap parameters
        self.L, self.M, self.N = self.gp.L, self.gp.M, self.gp.N
        self.Ms = tf.constant([0, self.M], dtype=INT())
        self.F = tf.transpose(tf.constant(self.gp.kernel.params.variance, dtype=FLOAT()))     # To convert (1,L) to (L,1)
        self.Lambda = tf.constant(self.gp.kernel.params.lengthscales, dtype=FLOAT())
        self.Lambda2 = self._Lambda2(is_diagonal=self.gp.kernel.is_independent)
        self.Lambda2_diag = self._Lambda2(is_diagonal=True)
        # Cache the training data kernel
        self.K_cho = tf.constant(self.gp.K_cho, dtype=FLOAT())
        self.K_inv_Y = tf.constant(self.gp.K_inv_Y, dtype=FLOAT())
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
            I = tf.eye(self.M, batch_shape=[1, 1, 1, 1], dtype=FLOAT())
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
        I = tf.eye(self.M, batch_shape=[1, 1], dtype=FLOAT())
        # First Moments
        G = tf.einsum('Mm, Llm, Nm -> LlNM', Theta, self.Lambda2[-1][1], self.gp.X)
        Phi = tf.einsum('Mm, Llm, Jm -> LlMJ', Theta, self.Lambda2[-1][1], Theta)
        Gamma = I - Phi
        # Second Moments
        Upsilon = tf.einsum('Mm, Llm, Llm, Jm -> LlMJ', Theta, self.Lambda2[1][1], self.Lambda2[-1][2], Theta)
        Gamma_inv = self.matrix_inverse(Gamma, I)
        print(tf.einsum('LlMm, LlmJ -> LlMJ', Gamma_inv, Gamma))
        Upsilon_inv = self.matrix_inverse(Upsilon, I)
        print(tf.einsum('LlMm, LlmJ -> LlMJ', Upsilon_inv, Upsilon))
        Pi = self.matrix_inverse(tf.einsum('LlMm, Llmj, LljJ -> LlMJ', Phi, Gamma_inv, Phi) + Upsilon_inv, I)

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

def magnitude(tensor: TF.Tensor, ein: str = 'lijk, lijk'):
    n = tf.cast(tf.reduce_prod(tensor.shape), FLOAT())
    return tf.sqrt(tf.divide(tf.einsum(ein, tensor, tensor), n))

def det(tensor: TF.Tensor):
    return tf.reduce_prod(tensor, axis=-1)
