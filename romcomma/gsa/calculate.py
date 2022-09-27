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

    # TWO_PI = tf.constant(2 * np.pi, dtype=FLOAT())
    # LOG_TWO_PI = tf.math.log(TWO_PI)

    @classmethod
    def det(cls, variance_cho):
        return tf.reduce_prod(variance_cho, axis=-1)

    @classmethod
    def pdf(cls, exponent: TF.Tensor, variance_cho_diagonal: TF.Tensor):
        """ Calculate the Gaussian pdf from the output of Gaussian.log_pdf.
        Args:
            exponent: The exponent in the Gaussian pdf.
            variance_cho_diagonal: The diagonal of the variance Cholesky decomposition.

        Returns: The Gaussian pdf.
        """
        return tf.exp(exponent) / Gaussian.det(variance_cho_diagonal)

    @classmethod
    def log_pdf(cls, mean: TF.Tensor, variance_cho: TF.Tensor, is_variance_diagonal: bool,
                ordinate: TF.Tensor = tf.constant(0, dtype=FLOAT()), LBunch: int = 2) -> Tuple[TF.Tensor, TF.Tensor]:
        """ Computes the logarithm of the un-normalized gaussian probability density, and the broadcast diagonal of variance_cho.
        Taking the product (Gaussian.det(variance_cho_diagonal) gives the normalization factor for the gaussian pdf.
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
        exponent = - 0.5 * tf.einsum('...o, ...o -> ...', exponent, exponent)
        return exponent, variance_cho


class ClosedIndex(gf.Module):
    """ Calculates closed Sobol Indices."""

    @classmethod
    @property
    def EFFECTIVELY_ZERO_COVARIANCE(cls) -> TF.Tensor:
        return tf.constant(1.0E-2, dtype=FLOAT())

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        """ Default calculation options.

        Returns: An empty dictionary.
        """
        return {}

    def _calculate(self):
        """ Called by constructor to calculate all available quantities prior to marginalization.
        These quantities suffice to calculate V[0], V[M].
        """
        pre_factor = Gaussian.det(tf.sqrt(self.Lambda2[1][0] * self.Lambda2[-1][1])) * self.F
        self.g0, _ = Gaussian.log_pdf(mean=self.gp.X[tf.newaxis, tf.newaxis, ...],
                                       variance_cho=tf.sqrt(self.Lambda2[1][1]), is_variance_diagonal=True, LBunch=2)
        self.g0 = pre_factor[..., tf.newaxis] * tf.exp(self.g0)     # Symmetric in L^2
        self.g0KY = self.g0 * self.K_inv_Y     # NOT symmetric in L^2
        self.KYg0_sum = tf.einsum('lLN -> l', self.g0KY)
        self.V = {'0': tf.einsum('l, j -> lj', self.KYg0_sum, self.KYg0_sum)}     # Symmetric in L^2
        self.G = tf.einsum('lLM, NM -> lLNM', self.Lambda2[-1][1], self.gp.X)     # Symmetric in L^2
        self.Phi = self.Lambda2[-1][1]     # Symmetric in L^2
        self.Gamma = 1 - self.Phi     # Symmetric in L^2
        self.V['M'] = self._V(self.G, self.Gamma)
        self.is_covariance_zero = tf.where(tf.abs(self.V['M']) > self.EFFECTIVELY_ZERO_COVARIANCE, tf.constant([False]), tf.constant([True]))
        self.V['M'] = tf.where(self.is_covariance_zero, tf.constant(1.0, dtype=FLOAT()), self.V['M'])

    def marginalize(self, m: TF.Slice) -> Dict[str, Dict[str: TF.Tensor]]:
        """ Calculate everything.
        Args:
            m: A Tf.Tensor pair of ints indicating the slice [m[0]:m[1]].
        Returns: The Sobol Index of m.
        """
        self._m = m

        G, Phi, Gamma = self.G, self.Phi, self.Gamma
        result = {'V': self._V(G[..., m[0]:m[1]], Gamma[..., m[0]:m[1]])}
        result['S'] = result['V'] / self.V['M']
        return result

    def _V(self, G: TF.Tensor, Gamma: TF.Tensor) -> TF.Tensor:
        """ Calculate V.

        Args:
            G: marginalized
            Gamma: marginalized
        Returns: V[m], according to marginalization.

        """
        Gamma_reshape = tf.expand_dims(Gamma, axis=2)
        Sigma = tf.expand_dims(Gamma_reshape, axis=2) + Gamma[tf.newaxis, tf.newaxis, ...]    # Symmetric in L^4
        Psi = Sigma - tf.einsum('lLM, jJM -> lLjJM', Gamma, Gamma)    # Symmetric in L^4
        SigmaPsi = tf.einsum('lLjJM, lLjJM -> lLjJM', Sigma, Psi)    # Symmetric in L^4
        SigmaG = tf.einsum('jJnM, lLNM -> lLNjJnM', Gamma_reshape, G) + tf.einsum('lLNM, jJnM -> lLNjJnM', Gamma_reshape, G)   # Symmetric in L^4 N^2
        # print(sym_check(SigmaG, [3, 4, 5, 0, 1, 2, 6])) note the symmetry.
        Sigma_pdf, Sigma_diag = Gaussian.log_pdf(mean=G, ordinate=G, variance_cho=tf.sqrt(Sigma), is_variance_diagonal=True, LBunch=2)
        SigmaPsi_pdf, SigmaPsi_diag = Gaussian.log_pdf(mean=SigmaG, variance_cho=tf.sqrt(SigmaPsi), is_variance_diagonal=True, LBunch=2)
        H = tf.exp(Sigma_pdf - SigmaPsi_pdf)   # Symmetric in L^4 N^2
        V = tf.einsum('lLN, lLNjJn, jJn -> lLjJ', self.g0KY, H, self.g0KY) / tf.sqrt(Gaussian.det(Psi))
        print(V)
        V = tf.einsum('lLjJ -> lj', V) - self.V['0']
        return V

    def _Lambda2(self) -> dict[int, Tuple[TF.Tensor]]:
        """ Calculate and cache the required powers of <Lambda^2 + J>.

        Returns: {1: <Lambda^2 + J>, -1: <Lambda^2 + J>^(-1)} for J in {0,1,2}.
        """
        if self.is_F_diagonal:
            result = tf.expand_dims(tf.einsum('lM, lM -> lM', self.Lambda, self.Lambda), axis=1)
        else:
            result = tf.einsum('lM, LM -> lLM', self.Lambda, self.Lambda)
            print(sym_check(result, [1, 0, 2]))
        result = tuple(result + j for j in range(3))
        return {1: result, -1: tuple(value**(-1) for value in result)}

    def __init__(self, gp: GPInterface, **kwargs: Any):
        """ Construct a ClosedIndex object. A wide range of values are collected or calculated and cached, especially via the final call to self._calculate.

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
        self.F = tf.constant(self.gp.kernel.params.variance, dtype=FLOAT())
        # Cache the training data kernel
        self.K_cho = tf.constant(self.gp.K_cho, dtype=FLOAT())
        self.K_inv_Y = tf.constant(self.gp.K_inv_Y, dtype=FLOAT())
        self.is_F_diagonal = not (gp._read_options() if gp._options_json.exists() else gp.OPTIONS).pop('kernel', {}).pop("variance", {}).pop('off_diagonal',
                                                                                                                                             False)
        if self.is_F_diagonal:
            self.F = self.F if self.F.shape[0] == 1 else tf.linalg.diag_part(self.F)
            self.F = tf.reshape(self.F, [self.L, 1])
        else:
            self.K_inv_Y = tf.transpose(self.K_inv_Y, [1, 0, 2])
        self.Lambda = tf.broadcast_to(tf.constant(self.gp.kernel.params.lengthscales, dtype=FLOAT()), [self.L, self.M])
        self.Lambda2 = self._Lambda2()
        # Calculate and cache values for m=0 and m=M
        self._calculate()


class ClosedIndexWithErrors(ClosedIndex):
    """ Calculates closed Sobol Indices with Errors."""
    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        """ Default calculation options. ``is_T_partial`` forces W[mM] = W[MM] = 0.

        Returns:
            is_T_calculated: If False, T and W are not calculated or returned.
            is_T_partial: If True this effectively asserts the full ['M'] model is variance free, so WmM is not calculated or returned.
        """
        return {'is_T_partial': True}

    def _calculate(self):
        """ Called by constructor to calculate all available quantities prior to marginalization.
        These quantities suffice to calculate V[0], V[M], A[00], self.A[m0]=A[M0] and self.A[mm]=A[MM]
        """
        super()._calculate()
        if self.is_F_diagonal:
            self.Upsilon = self.Lambda2[1][1] * self.Lambda2[-1][2]
            self.V2MM = self.V['M'] * self.V['M']
            self.mu_phi_mu = {'pre-factor': tf.sqrt(Gaussian.det(self.Lambda2[1][0] * self.Lambda2[-1][2])) * self.F}
            self.mu_phi_mu['pre-factor'] = tf.transpose(self.mu_phi_mu['pre-factor'], [1, 0])
            self.G_log_pdf = Gaussian.log_pdf(mean=self.G, variance_cho=tf.sqrt(self.Phi), is_variance_diagonal=True, LBunch=2)
            self.Upsilon_log_pdf = self._Upsilon_log_pdf(self.G, self.Phi, self.Upsilon)
            self.Omega_log_pdf = self._Omega_log_pdf(self.Ms, self.Ms, self.G, self.Phi, self.Gamma, self.Upsilon)
            factor = tf.einsum('k, jJN -> kjJN', self.KYg0_sum, self.g0)
            self.psi_factor = {'shape': factor.shape[:-2].as_list() + [-1, 1]}
            self.psi_factor['0'] = tf.squeeze(tf.linalg.triangular_solve(self.K_cho, tf.reshape(factor, self.psi_factor['shape'])), axis=-1)
            self.psi_factor['M'] = self._psi_factor(self.G, self.Phi, self.G_log_pdf)
            A_mislabelled = self._A(self._mu_phi_mu(self.G_log_pdf, self.Upsilon_log_pdf, self.Omega_log_pdf, self.Omega_log_pdf, is_constructor=True),
                                    self._mu_psi_mu(self.psi_factor['M'], is_constructor=True))
            self.A = {'00': A_mislabelled.pop('Mm')}
            if not self.options['is_T_partial']:
                self.W = self._W(**A_mislabelled)
                self.A = self.A | A_mislabelled

    def marginalize(self, m: TF.Slice) -> Dict[str, Dict[str: TF.Tensor]]:
        """ Calculate everything.
        Args:
            m: A Tf.Tensor pair of ints indicating the slice [m[0]:m[1]].
        Returns: The Sobol Index of m, with errors (T and W).
        """
        result = super().marginalize(m)
        if self.is_F_diagonal:
            G, Phi, Gamma = self.G, self.Phi, self.Gamma
            Upsilon = self.Upsilon
            G_m = G[..., m[0]:m[1]]
            Phi_mm = Phi[..., m[0]:m[1]]
            G_log_pdf = Gaussian.log_pdf(G_m, tf.sqrt(Phi_mm), is_variance_diagonal=True, LBunch=2)
            Upsilon_log_pdf = self._Upsilon_log_pdf(G_m, Phi_mm, Upsilon[..., m[0]:m[1]])
            Omega_log_pdf_M = self._Omega_log_pdf(self.Ms, m, G, Phi, Gamma, Upsilon)
            Omega_log_pdf_m = self._Omega_log_pdf(m, m, G, Phi, Gamma, Upsilon)
            psi_factor = self._psi_factor(G[..., m[0]:m[1]], Phi[..., m[0]:m[1]], G_log_pdf)
            result = result | self._T(result['V'],
                                      **self._W(**self._A(self._mu_phi_mu(G_log_pdf, Upsilon_log_pdf, Omega_log_pdf_M, Omega_log_pdf_m),
                                                          self._mu_psi_mu(psi_factor))))
            return result

    def _T(self, Vm: TF.Tensor, mm: TF.Tensor, Mm: TF.Tensor = TF.NOT_CALCULATED) -> Dict[str, TF.Tensor]:
        """ Calculate T

        Args:
            Vm:
            mm: W[mm]
            Mm: W[Mm]

        Returns: T[mm]

        """
        T = mm
        if not self.options['is_T_partial']:
            V_ratio = Vm / self.V['M']
            T += self.W['mm'] * V_ratio * V_ratio - 2 * Mm * V_ratio
        return {'T': T / self.V2MM, 'Wmm': mm} if self.options['is_T_partial'] else {'T': T / self.V2MM, 'Wmm': mm, 'WmM': Mm}

    def _W(self, Om: TF.Tensor, m0: TF.Tensor, mm: TF.Tensor, Mm: TF.Tensor = TF.NOT_CALCULATED) -> Dict[str, TF.Tensor]:
        """ Calculate W.

        Args:
            Om: A[0m]
            m0: A[m0]
            mm: A[mm]
            Mm: A[Mm]
        Returns: W[mm] if is_T_partial, else W{mm, Mm}
        """
        W = {'mm': mm - m0 - Om + self.A['00']}
        if not self.options['is_T_partial'] and Mm.dtype.is_floating:
            W['Mm'] = Mm - self.A['m0'] - Om + self.A['00']
        return W

    def _A(self, mu_phi_mu: TF.Tensor, mu_psi_mu: TF.Tensor) -> Dict[str, TF.Tensor]:
        A = mu_phi_mu - mu_psi_mu
        # A = tf.concat([tf.transpose(A[0:1, ...], [0, 4, 3, 2, 1]), A], axis=0)  # Create Om from m0 symmetry of mu_phi_mu and mu_psi_mu.
        A += tf.transpose(A, [0, 2, 1])
        A *= 2
        return {'Om': A[0], 'm0': A[1], 'mm': A[2], 'Mm': A[-1]} if A.shape[0] > 2 else {'Mm': A[0]}

    def _mu_psi_mu(self, psi_factor: TF.Tensor, is_constructor: bool = False) -> TF.Tensor:
        """

        Args:
            psi_factor:
            is_constructor:
        Returns:
        """
        mu_psi_mu = []
        if self.gp.kernel.is_independent:
            if not is_constructor or (is_constructor and not self.options['is_T_partial']):
                mu_psi_mu += [tf.einsum('liS, liS -> li', psi_factor, self.psi_factor['0']),
                              tf.einsum('liS, liS -> li', psi_factor, psi_factor)]
            if is_constructor:
                mu_psi_mu += [tf.einsum('liS, liS -> li', self.psi_factor['0'], self.psi_factor['0'])]
            else:
                if not self.options['is_T_partial']:
                    mu_psi_mu += [tf.einsum('liS, liS -> li', self.psi_factor['M'], psi_factor)]
            mu_psi_mu = [tf.transpose(tf.linalg.diag(item), [0, 2, 3, 1]) for item in mu_psi_mu]
        else:
            if not is_constructor or (is_constructor and not self.options['is_T_partial']):
                mu_psi_mu += [tf.einsum('liS, kjS -> lijk', psi_factor, self.psi_factor['0']),
                              tf.einsum('liS, kjS -> lijk', psi_factor, psi_factor)]
            if is_constructor:
                mu_psi_mu += [tf.einsum('liS, kjS -> lijk', self.psi_factor['0'], self.psi_factor['0'])]
            else:
                if not self.options['is_T_partial']:
                    mu_psi_mu += [tf.einsum('liS, kjS -> lijk', self.psi_factor['M'], psi_factor)]
        # FIXME: Debug
        # if len(mu_psi_mu) > 1 and self._m[1]==1:
        #     print(self.gp.kernel.is_independent)
        #     for l in (0, 1):
        #         for i in (0, 1):
        #             if self.options['is_T_diagonal']:
        #                 print(f'mu_psi_mu[{l},{i}]={mu_psi_mu[1][l,i,0,0]}')
        #             else:
        #                 print(f'mu_psi_mu[{l},{i}]={mu_psi_mu[1][l, i, l, i]}')
        return tf.stack(mu_psi_mu)

    def _psi_factor(self, G: TF.Tensor, Phi: TF.Tensor, G_log_pdf: Tuple[TF.Tensor, TF.Tensor]) -> TF.Tensor:
        """ Calculate the psi factor E_mp such that E[m] E[mp] (mu[m] psi[m][mp] mu[mp]) = E_m E_mp

        Args:
            G: jJNm
            Phi: kKm
            G_log_pdf: kKn
        Returns: kjS
        """
        D = Phi[..., tf.newaxis, tf.newaxis, :] - tf.einsum('kKM, jJM, kKM -> kKjJM', Phi, Phi, Phi)
        mean = tf.einsum('kKM, jJNM -> kKjJNM', Phi, G)
        mean = mean[:, :, tf.newaxis, ...] - G[..., tf.newaxis, tf.newaxis, tf.newaxis, :]
        log_pdf = list(Gaussian.log_pdf(mean=mean, variance_cho=tf.sqrt(D), is_variance_diagonal=True, LBunch=2))
        log_pdf[0] -= G_log_pdf[0][..., tf.newaxis, tf.newaxis, tf.newaxis]
        log_pdf[1] /= G_log_pdf[1][..., tf.newaxis, tf.newaxis, tf.newaxis, :]
        factor = tf.einsum('kKn, jJN, kKnjJN -> kjJN', self.g0KY, self.g0, Gaussian.pdf(*tuple(log_pdf)))
        factor = tf.squeeze(tf.linalg.triangular_solve(self.K_cho[tf.newaxis, ...], tf.reshape(factor, self.psi_factor['shape'])), axis=-1)
        return factor

    def _mu_phi_mu(self, G_log_pdf: TF.Tensor, Upsilon_log_pdf: TF.Tensor, Omega_log_pdf_M: TF.Tensor, Omega_log_pdf_m: TF.Tensor,
                   is_constructor: bool = False) -> TF.Tensor:
        """ Calculate E[m] E[mp] (mu[m] phi[m][mp] mu[mp]).

        Args:
            G_log_pdf: kKN
            Upsilon_log_pdf: liLNj
            Omega_log_pdf_M: liLNjkKn
            Omega_log_pdf_m: liLNjkKn
            is_constructor:

        Returns: lijk
        """
        mu_phi_mu = []
        Omega_log_pdf_m = list(Omega_log_pdf_m)
        Omega_log_pdf_M = list(Omega_log_pdf_M)
        if not is_constructor or (is_constructor and not self.options['is_T_partial']):
            Omega_log_pdf_m[0] += Upsilon_log_pdf[0][..., tf.newaxis, tf.newaxis, tf.newaxis] - G_log_pdf[0]
            Omega_log_pdf_m[1] *= Upsilon_log_pdf[1][..., tf.newaxis, tf.newaxis, tf.newaxis, :] / G_log_pdf[1]
            mu_phi_mu += [tf.einsum('lLN, liLNj, k -> lijk', self.g0KY, Gaussian.pdf(*Upsilon_log_pdf), self.KYg0_sum),
                          tf.einsum('lLN, liLNjkKn, kKn -> lijk', self.g0KY, Gaussian.pdf(*tuple(Omega_log_pdf_m)), self.g0KY)]
        if is_constructor:
            mu_phi_mu += [tf.expand_dims(tf.expand_dims(tf.einsum('l, k -> lk', self.KYg0_sum, self.KYg0_sum), axis=1), axis=1)]
        else:
            if not self.options['is_T_partial']:
                Omega_log_pdf_M[0] -= G_log_pdf[0]
                Omega_log_pdf_M[1] /= G_log_pdf[1]
                pdf = Gaussian.pdf(*tuple(Omega_log_pdf_M)) * Gaussian.pdf(*self.Upsilon_log_pdf)[..., tf.newaxis, tf.newaxis, tf.newaxis]
                mu_phi_mu += [tf.einsum('lLN, liLNjkKn, kKn -> lijk', self.g0KY, pdf, self.g0KY)]
        mu_phi_mu = [item * self.mu_phi_mu['pre-factor'] for item in mu_phi_mu]
        # FIXME: Debug
        # if len(mu_phi_mu) > 1 and self._m[1]==1:
        #     print(self.gp.kernel.is_independent)
        #     print(self.mu_phi_mu['pre-factor'])
        #     for l in (0, 1):
        #         for i in (0, 1):
        #             if self.options['is_T_diagonal']:
        #                 print(f'mu_phi_mu[{l},{i}]={mu_phi_mu[1][l,i,0,0]}')
        #             else:
        #                 print(f'mu_phi_mu[{l},{i}]={mu_phi_mu[1][l, i, l, i]}')
        return tf.stack(mu_phi_mu)

    def _Omega_log_pdf(self, m: TF.Slice, mp: TF.Slice, G: TF.Tensor, Phi: TF.Tensor, Gamma: TF.Tensor, Upsilon: TF.Tensor) -> Tuple[TF.Tensor, TF.Tensor]:
        """ The Omega integral for m=mp or m=[:M]. Does not apply when m=[0:0].

        Args:
            m: The marginalization m.
            mp: The marginalization m_primed.
            G: Un-marginalized. lLNM and kKnM.
            Phi: Un-marginalized. ijM and kKM.
            Gamma: Un-marginalized. lLM, ijM.
            Upsilon: Un-marginalized. ijM.
        Returns: liLNjkKn.
        """
        Gamma_inv = 1 / Gamma
        Pi = 1 + Phi + tf.einsum('ijM, ijM, ijM -> ijM', Phi, Gamma_inv, Phi)
        Pi = 1 / Pi
        B = tf.einsum('kKM, kKM -> kKM', 1 - Phi, Phi)[tf.newaxis, tf.newaxis, ...]
        B += tf.einsum('kKM, ijM, kKM -> ijkKM', Phi, Pi, Phi)
        Gamma_reshape = tf.expand_dims(tf.expand_dims(Gamma, axis=1), axis=3)
        C = Gamma_reshape / (Gamma_reshape + tf.einsum('lLM, ijM -> liLjM', Phi, Upsilon))
        C = tf.einsum('ijM, liLjM -> liLjM', Upsilon, C)
        Omega = tf.einsum('ijM, ijM, ijM -> ijM', Pi, Phi, Gamma_inv)
        Omega = tf.einsum('kKM, ijM -> ijkKM', Phi, Omega)
        mean = tf.einsum('ijkKM, liLjM, lLM, lLNM -> liLNjkKM', Omega, C, Gamma_inv, G)
        if m[1] < mp[1]:
            pad = [m[0], self.M - m[1]]
            mean = tf.pad(mean[..., m[0]:m[1]], pad)
            variance = (tf.expand_dims(tf.expand_dims(B, axis=0), axis=2) +
                        tf.pad(tf.einsum('ijkKM, liLjM, ijkKM -> liLjkKM', Omega, C, Omega)[..., m[0]:m[1]], pad))

        else:
            variance = tf.expand_dims(tf.expand_dims(B, axis=0), axis=2) + tf.einsum('ijkKM, liLjM, ijkKM -> liLjkKM', Omega, C, Omega)
        if mp is not self.Ms:
            variance = variance[..., mp[0]:mp[1]]
            mean = mean[..., mp[0]:mp[1]]
            G = G[..., mp[0]:mp[1]]
        mean = tf.expand_dims(mean, axis=-2) - G[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]
        return Gaussian.log_pdf(mean=mean, variance_cho=tf.sqrt(variance), is_variance_diagonal=True, LBunch=3)

    def _Upsilon_log_pdf(self, G: TF.Tensor, Phi: TF.Tensor, Upsilon: TF.Tensor) -> Tuple[TF.Tensor, TF.Tensor]:
        """ The Upsilon integral.

        Args:
            G: lLNM
            Phi: lLM
            Upsilon: ijM
        Returns: liLNj
        """
        sqrt_1_Upsilon = tf.sqrt(1 - Upsilon)
        mean = tf.einsum('ijM, lLNM -> liLNjM', sqrt_1_Upsilon, G)
        variance = 1 - tf.einsum('ijM, lLM, ijM -> liLjM', sqrt_1_Upsilon, Phi, sqrt_1_Upsilon)
        return Gaussian.log_pdf(mean, tf.sqrt(variance), is_variance_diagonal=True, LBunch=3)


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


def mean(tensor: TF.Tensor):
    n = tf.cast(tf.reduce_prod(tensor.shape), FLOAT())
    return tf.divide(tf.reduce_sum(tensor), n)


def sos(tensor: TF.Tensor, ein: str = 'lijk, lijk'):
    return tf.einsum(ein, tensor, tensor)


def ms(tensor: TF.Tensor, ein: str = 'lijk'):
    n = tf.cast(tf.reduce_prod(tensor.shape), FLOAT())
    return tf.divide(sos(tensor, ein), n)


def rms(tensor: TF.Tensor, ein: str = 'lijk, lijk'):
    return tf.sqrt(ms(tensor, ein))


def det(tensor: TF.Tensor):
    return tf.reduce_prod(tensor, axis=-1)

I = [0, 0, 0, 0]
