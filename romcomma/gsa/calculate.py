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
from enum import IntEnum


LogPDF = Tuple[TF.Tensor, TF.Tensor]


class Gaussian(ABC):
    """ Encapsulates the calculation of a Gaussian pdf. Not instantiatable."""

    # TWO_PI = tf.constant(2 * np.pi, dtype=FLOAT())
    # LOG_TWO_PI = tf.math.log(TWO_PI)

    @classmethod
    def det(cls, variance_cho_diagonal):
        return tf.reduce_prod(variance_cho_diagonal, axis=-1)

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
                ordinate: TF.Tensor = tf.constant(0, dtype=FLOAT()), LBunch: int = 2) -> LogPDF:
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
            exponent = ordinate / tf.broadcast_to(variance_cho, tf.concat([variance_cho.shape[:-2], ordinate.shape[-2:]], axis=0))
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

    def marginalize(self, m: TF.Slice) -> Dict[str, Dict[str: TF.Tensor]]:
        """ Calculate everything.
        Args:
            m: A Tf.Tensor pair of ints indicating the slice [m[0]:m[1]].
        Returns: The Sobol Index of m.
        """
        self._m = m

        G, Phi = self.G, self.Phi
        result = {'V': self._V(G[..., m[0]:m[1]], Phi[..., m[0]:m[1]])}
        result['S'] = result['V'] / self.V['M']
        return result

    def _V(self, G: TF.Tensor, Phi: TF.Tensor) -> TF.Tensor:
        """ Calculate V.

        Args:
            G: marginalized
            Gamma: marginalized
        Returns: V[m], according to marginalization.

        """
        Gamma = 1 - Phi
        Psi = tf.expand_dims(tf.expand_dims(Gamma, axis=2), axis=2) + Gamma[tf.newaxis, tf.newaxis, ...]    # Symmetric in L^4
        Psi = Psi - tf.einsum('lLM, jJM -> lLjJM', Gamma, Gamma)    # Symmetric in L^4
        PsiPhi = tf.einsum('lLjJM, lLM -> lLjJM', Psi, Phi)    # Symmetric in L^4
        PhiG = tf.expand_dims(tf.einsum('lLM, jJnM -> lLjJnM', Phi, G), axis=2)    # Symmetric in L^4 N^2
        # print(sym_check(PhiG, [3, 4, 5, 0, 1, 2, 6])) note the symmetry.
        Phi_pdf, Phi_diag = Gaussian.log_pdf(mean=G, variance_cho=tf.sqrt(Phi), is_variance_diagonal=True, LBunch=2)
        Psi_pdf, Psi_diag = Gaussian.log_pdf(mean=PhiG, variance_cho=tf.sqrt(PsiPhi), ordinate=G[..., tf.newaxis, tf.newaxis, tf.newaxis, :],
                                             is_variance_diagonal=True, LBunch=2)
        H = tf.exp(Psi_pdf - Phi_pdf[..., tf.newaxis, tf.newaxis, tf.newaxis])   # Symmetric in L^4 N^2
        # print(sym_check(H, [0, 1, 2, 4, 3, 5])) note the symmetry.
        V = tf.einsum('lLN, lLNjJn, jJn -> lLjJ', self.g0KY, H, self.g0KY) / tf.sqrt(Gaussian.det(Psi))         # Only one symmetry in L^4
        # print(sym_check(V, [2, 3, 0, 1])) note the symmetry.
        V = tf.einsum('lLjJ -> lj', V) - self.V['0']        # Symmetric in L^2
        return V

    def _calculate(self):
        """ Called by constructor to calculate all available quantities prior to marginalization.
        These quantities suffice to calculate V[0], V[M].
        """
        pre_factor = tf.sqrt(Gaussian.det(self.Lambda2[1][0] * self.Lambda2[-1][1])) * self.F
        self.g0, _ = Gaussian.log_pdf(mean=self.gp.X[tf.newaxis, tf.newaxis, ...],
                                       variance_cho=tf.sqrt(self.Lambda2[1][1]), is_variance_diagonal=True, LBunch=2)
        self.g0 = pre_factor[..., tf.newaxis] * tf.exp(self.g0)     # Symmetric in L^2
        self.g0KY = self.g0 * self.K_inv_Y     # NOT symmetric in L^2
        self.g0KY_sum = tf.einsum('lLN -> l', self.g0KY)
        self.V = {'0': tf.einsum('l, j -> lj', self.g0KY_sum, self.g0KY_sum)}     # Symmetric in L^2
        self.G = tf.einsum('lLM, NM -> lLNM', self.Lambda2[-1][1], self.gp.X)     # Symmetric in L^2
        self.Phi = self.Lambda2[-1][1]     # Symmetric in L^2
        self.V['M'] = self._V(self.G, self.Phi)
        self.is_covariance_zero = tf.where(tf.abs(self.V['M']) > self.EFFECTIVELY_ZERO_COVARIANCE, tf.constant([False]), tf.constant([True]))
        self.V['M'] = tf.where(self.is_covariance_zero, tf.constant(1.0, dtype=FLOAT()), self.V['M'])

    def _Lambda2(self) -> dict[int, Tuple[TF.Tensor]]:
        """ Calculate and cache the required powers of <Lambda^2 + J>.

        Returns: {1: <Lambda^2 + J>, -1: <Lambda^2 + J>^(-1)} for J in {0,1,2}.
        """
        if self.is_F_diagonal:
            result = tf.einsum('lM, lM -> lM', self.Lambda, self.Lambda)[:, tf.newaxis, :]
        else:
            result = tf.einsum('lM, LM -> lLM', self.Lambda, self.Lambda)
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
        # Determine if F is diagonal
        self.is_F_diagonal = self.options.pop('is_F_diagonal', None)
        if self.is_F_diagonal is None:
            gp_options = self.gp._read_options() if self.gp._options_json.exists() else self.gp.OPTIONS
            self.is_F_diagonal = not gp_options.pop('kernel', {}).pop("variance", {}).pop('off_diagonal', False)
        # Reshape according to is_F_diagonal
        if self.is_F_diagonal:
            self.F = self.F if self.F.shape[0] == 1 else tf.linalg.diag_part(self.F)
            self.F = tf.reshape(self.F, [self.L, 1])
        else:
            self.K_inv_Y = tf.transpose(self.K_inv_Y, [1, 0, 2])
        # Set Lambdas
        self.Lambda = tf.broadcast_to(tf.constant(self.gp.kernel.params.lengthscales, dtype=FLOAT()), [self.L, self.M])
        self.Lambda2 = self._Lambda2()
        # Calculate and store values for m=0 and m=M
        self._calculate()


class ClosedIndexWithErrors(ClosedIndex):
    """ Calculates closed Sobol Indices with Errors."""

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        """ Default calculation options. ``is_T_partial`` forces W[mM] = W[MM] = 0.

        Returns:
            is_T_partial: If True this effectively asserts the full ['M'] model is variance free, so WmM is not calculated or returned.
        """
        return {'is_T_partial': True}

    class EquatedRanks(NamedTuple):
        l_j_and_i_k: Any
        l_k_and_i_j: Any

    class MangledEinstring(NamedTuple):
        l: str
        i: str
        j: str
        k: str

    _EIN: EquatedRanks[MangledEinstring] = EquatedRanks(l_j_and_i_k=MangledEinstring(l='j', i='k', j='l', k='i'),
                                                        l_k_and_i_j=MangledEinstring(l='k', i='j', j='i', k='l'))

    def equate_ranks(self, liLNjkJM: TF.Tensor, ein: MangledEinstring) -> TF.Tensor:
        """

        Args:
            liLNjkJM: A tensor which must have ranks liLNjkJM
            ein: A top-level element of ClosedIndexWithErrors._EIN.

        Returns:
            liLNjkJM with ranks equated.
        """
        shape = liLNjkJM.shape.as_list()
        ein_j = 'j' if shape[4] == 1 else ein.j
        ein_k = 'k' if shape[5] == 1 else ein.k
        liLNjkJM = tf.reshape(liLNjkJM, shape[:-2] + [-1])
        result = tf.einsum(f'liLN{ein_j}{ein_k}S -> LN{ein.j}{ein.k}S', liLNjkJM)
        result = tf.reshape(result, result.shape[:-1].as_list() + shape[-2:])
        return tf.einsum(f'LNjjJM -> LNjJM', result)[..., tf.newaxis, :, :] if self.is_F_diagonal and ein.j == 'i' else result

    def equated_ranks_gaussian_log_pdf(self, mean: TF.Tensor, variance: TF.Tensor, ordinate: TF.Tensor) -> EquatedRanks[LogPDF]:
        variance_cho = tf.sqrt(variance)
        result = []
        N_axis = 3
        for ein in self._EIN:
            equated_ranks_variance_cho = self.equate_ranks(tf.expand_dims(variance_cho, N_axis), ein)[..., tf.newaxis, :]
            equated_ranks_mean = self.equate_ranks(mean, ein)[..., tf.newaxis, :]
            shape = tf.concat([equated_ranks_mean.shape[:-2], ordinate.shape[-2:]], axis=0) if tf.rank(ordinate) > 2 else None
            equated_ranks_mean = (equated_ranks_mean if shape is None else tf.broadcast_to(equated_ranks_mean, shape)) - ordinate
            result += [Gaussian.log_pdf(mean=equated_ranks_mean, variance_cho=equated_ranks_variance_cho, is_variance_diagonal=True, LBunch=10000)]
        return self.EquatedRanks._make(result)

    def _Omega_log_pdf(self, m: TF.Slice, mp: TF.Slice, G: TF.Tensor, Phi: TF.Tensor, Upsilon: TF.Tensor) -> EquatedRanks[LogPDF]:
        """ The Omega integral for m=mp or m=[:M]. Does not apply when m=[0:0].

        Args:
            m: The marginalization m.
            mp: The marginalization m_primed.
            G: Un-marginalized. lLNM and jJnM.
            Phi: Un-marginalized. ikM and jJM.
            Upsilon: Un-marginalized. ikM.
        Returns: liLNjkJn.
        """
        Gamma = 1 - Phi
        Gamma_inv = 1 / Gamma
        Pi = 1 + Phi + tf.einsum('ikM, ikM, ikM -> ikM', Phi, Gamma_inv, Phi)
        Pi = 1 / Pi
        B = tf.einsum('jJM, jJM -> jJM', Gamma, Phi)[tf.newaxis, :, tf.newaxis, ...]
        B += tf.einsum('jJM, ikM, jJM -> ijkJM', Phi, Pi, Phi)
        Gamma_reshape = Gamma[:, tf.newaxis, :, tf.newaxis, :]
        C = Gamma_reshape / (1 - tf.einsum('lLM, ikM -> liLkM', Phi, Upsilon))
        C = tf.einsum('ikM, liLkM -> liLkM', (1 - Upsilon), C)
        Omega = tf.einsum('ikM, ikM, ikM -> ikM', Pi, Phi, Gamma_inv)
        Omega = tf.einsum('jJM, ikM -> ijkJM', Phi, Omega)
        mean = tf.einsum('ijkJM, liLkM, lLM, lLNM -> liLNjkJM', Omega, C, Gamma_inv, G)
        variance = B[tf.newaxis, :, tf.newaxis, ...] + tf.einsum('ijkJM, liLkM, ijkJM -> liLjkJM', Omega, C, Omega)
        if mp is not self.Ms:
            variance = variance[..., mp[0]:mp[1]]
            mean = mean[..., mp[0]:mp[1]]
            G = G[..., mp[0]:mp[1]]
        return self.equated_ranks_gaussian_log_pdf(mean, variance, G[:, tf.newaxis, ...])

    def _Upsilon_log_pdf(self, G: TF.Tensor, Phi: TF.Tensor, Upsilon: TF.Tensor) -> EquatedRanks[LogPDF]:
        """ The Upsilon integral.

        Args:
            G: lLNM
            Phi: lLM
            Upsilon: ikM
        Returns: liLNjkJn
        """
        Upsilon_cho = tf.sqrt(Upsilon)
        mean = tf.einsum('ikM, lLNM -> liLNkM', Upsilon_cho, G)[..., tf.newaxis, :, tf.newaxis, :]
        variance = 1 - tf.einsum('ikM, lLM, ikM -> liLkM', Upsilon_cho, Phi, Upsilon_cho)[..., tf.newaxis, :, tf.newaxis, :]
        return self.equated_ranks_gaussian_log_pdf(mean, variance, tf.constant(0, dtype=FLOAT()))

    def _mu_phi_mu(self, G_log_pdf: TF.Tensor, Upsilon_log_pdf: TF.Tensor, Omega_log_pdf_M: TF.Tensor, Omega_log_pdf_m: TF.Tensor,
                   is_constructor: bool = False) -> TF.Tensor:
        """ Calculate E[m] E[mp] (mu[m] phi[m][mp] mu[mp]).

        Args:
            G_log_pdf: jJn
            Upsilon_log_pdf: liLNjk
            Omega_log_pdf_M: liLNjkJn
            Omega_log_pdf_m: liLNjkJn
            is_constructor:

        Returns: li
        """
        G_log_pdf = [g[:, tf.newaxis, ...] for g in G_log_pdf]
        mu_phi_mu = [[], []]
        Omega_log_pdf_m = list(Omega_log_pdf_m)
        Omega_log_pdf_M = list(Omega_log_pdf_M)
        for i, ein in enumerate(self._EIN):
            ein_j, ein_jJ, ein_0 = ((f'jLN, LNjkJn, j -> j', f'jLN, LNjkJn, jJn -> j', f'l, l -> l') if self.is_F_diagonal and ein.l == 'k' else
                                    (f'{ein.l}LN, LNjkJn, j -> {ein.l}{ein.i}', f'{ein.l}LN, LNjkJn, jJn -> {ein.l}{ein.i}', f'l, i -> li'))
            Omega_log_pdf_m[i] = list(Omega_log_pdf_m[i])
            Omega_log_pdf_M[i] = list(Omega_log_pdf_M[i])
            if not is_constructor or (is_constructor and not self.options['is_T_partial']):
                Omega_log_pdf_m[i][0] += Upsilon_log_pdf[i][0] - G_log_pdf[0]
                Omega_log_pdf_m[i][1] *= Upsilon_log_pdf[i][1] / G_log_pdf[1]
                pdf = Gaussian.pdf(*Upsilon_log_pdf[i])
                mu_phi_mu[i] += [tf.einsum(ein_j, self.g0KY, pdf, self.g0KY_sum),
                                 tf.einsum(ein_jJ, self.g0KY, Gaussian.pdf(*tuple(Omega_log_pdf_m[i])), self.g0KY)]
            if is_constructor:
                mu_phi_mu[i] += [tf.einsum(ein_0, self.g0KY_sum, self.g0KY_sum)]
            else:
                if not self.options['is_T_partial']:
                    Omega_log_pdf_M[i][0] -= G_log_pdf[0]
                    Omega_log_pdf_M[i][1] /= G_log_pdf[1]
                    pdf = Gaussian.pdf(*tuple(Omega_log_pdf_M[i])) * Gaussian.pdf(*self.Upsilon_log_pdf[i])[..., tf.newaxis, tf.newaxis, tf.newaxis]
                    mu_phi_mu += [tf.einsum(ein_jJ, self.g0KY, pdf, self.g0KY)]
        mu_phi_mu = self.EquatedRanks._make(mu_phi_mu)
        if self.is_F_diagonal:
            mu_phi_mu = [tf.linalg.diag(tf.einsum('i, i -> i', self.mu_phi_mu['pre-factor'], mu_phi_mu.l_k_and_i_j[index]))
                         + tf.einsum('i, li -> li', self.mu_phi_mu['pre-factor'], mu_phi_mu.l_j_and_i_k[index])
                         for index in range(len(mu_phi_mu.l_j_and_i_k))]
        else:
            mu_phi_mu = [tf.einsum('il, li -> li', self.mu_phi_mu['pre-factor'], mu_phi_mu.l_k_and_i_j[index])
                         + tf.einsum('ii, li -> li', self.mu_phi_mu['pre-factor'], mu_phi_mu.l_j_and_i_k[index])
                         for index in range(len(mu_phi_mu.l_j_and_i_k))]
        return tf.stack(mu_phi_mu)

    def _psi_factor_reshape(self, factor):
        if tf.rank(self.K_cho) == 2:
            if factor.shape[-2] == 1:
                factor = tf.einsum('lNiI -> liIN', tf.linalg.diag(tf.einsum('liIN -> lNi', factor)))
        return tf.reshape(factor, factor.shape[:-2].as_list() + [-1, 1])

    def _psi_factor(self, G: TF.Tensor, Phi: TF.Tensor, G_log_pdf: Tuple[TF.Tensor, TF.Tensor]) -> TF.Tensor:
        """ Calculate the psi factor E_mp such that E[m] E[mp] (mu[m] psi[m][mp] mu[mp]) = E_m E_mp

        Args:
            G: lLNm
            Phi: lLm
            G_log_pdf: lLn
        Returns: liS
        """
        D = Phi[..., tf.newaxis, tf.newaxis, :] - tf.einsum('lLM, iIM, lLM -> lLiIM', Phi, Phi, Phi)
        mean = tf.einsum('lLM, iInM -> lLiInM', Phi, G)
        mean = mean[:, :, tf.newaxis, ...] - G[..., tf.newaxis, tf.newaxis, tf.newaxis, :]
        log_pdf = list(Gaussian.log_pdf(mean=mean, variance_cho=tf.sqrt(D), is_variance_diagonal=True, LBunch=2))
        log_pdf[0] -= G_log_pdf[0][..., tf.newaxis, tf.newaxis, tf.newaxis]
        log_pdf[1] /= G_log_pdf[1][..., tf.newaxis, tf.newaxis, tf.newaxis, :]
        factor = tf.einsum('lLN, iIn, lLNiIn -> liIn', self.g0KY, self.g0, Gaussian.pdf(*tuple(log_pdf)))
        factor = tf.squeeze(tf.linalg.triangular_solve(self.K_cho, self._psi_factor_reshape(factor)), axis=-1)
        return factor

    def _mu_psi_mu(self, psi_factor: TF.Tensor, is_constructor: bool = False) -> TF.Tensor:
        """

        Args:
            psi_factor: liS
            is_constructor: For initial singular call is self._calculate.
        Returns: li
        """
        mu_psi_mu = []
        if self.is_F_diagonal:
            def sym_prod(psi_fac_0: TF.Tensor, psi_fac_1: TF.Tensor):
                result = tf.einsum('liS, liS -> li', psi_fac_0, psi_fac_1)
                return tf.linalg.set_diag(result, 2 * tf.linalg.diag_part(result))
        else:
            def sym_prod(psi_fac_0: TF.Tensor, psi_fac_1: TF.Tensor):
                return tf.einsum('liS, ilS -> li', psi_fac_0, psi_fac_1) + tf.einsum('liS, liS -> li', psi_fac_0, psi_fac_1)

        if not is_constructor or (is_constructor and not self.options['is_T_partial']):
            mu_psi_mu += [sym_prod(psi_factor, self.psi_factor['0']), sym_prod(psi_factor, psi_factor)]
        if is_constructor:
            mu_psi_mu += [sym_prod(self.psi_factor['0'], self.psi_factor['0'])]
        else:
            if not self.options['is_T_partial']:
                mu_psi_mu += [sym_prod(self.psi_factor['M'], psi_factor)]
        return tf.stack(mu_psi_mu)

    def _A(self, mu_phi_mu: TF.Tensor, mu_psi_mu: TF.Tensor) -> Dict[str, TF.Tensor]:
        A = mu_phi_mu - mu_psi_mu
        A += tf.transpose(A, [0, 2, 1])
        return {'Om': A[0], 'm0': A[0], 'mm': A[1], 'Mm': A[-1]} if A.shape[0] > 1 else {'Mm': A[0]}
        # return {'OO': self.A['00'], 'm0': A[0], 'mm': A[1], 'Mm_': A[-1]} if A.shape[0] > 1 else {'Mm': A[0]}

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
        return {'T': T / self.V2MM, 'Wmm': mm} if self.options['is_T_partial'] else {'T': T / self.V2MM, 'Wmm': mm, 'WmM_': Mm}

    def marginalize(self, m: TF.Slice) -> Dict[str, Dict[str: TF.Tensor]]:
        """ Calculate everything.
        Args:
            m: A Tf.Tensor pair of ints indicating the slice [m[0]:m[1]].
        Returns: The Sobol Index of m, with errors (T and W).
        """
        result = super().marginalize(m)
        G, Phi = self.G, self.Phi
        Upsilon = self.Upsilon
        G_m = G[..., m[0]:m[1]]
        Phi_mm = Phi[..., m[0]:m[1]]
        G_log_pdf = Gaussian.log_pdf(G_m, tf.sqrt(Phi_mm), is_variance_diagonal=True, LBunch=2)
        Upsilon_log_pdf = self._Upsilon_log_pdf(G_m, Phi_mm, Upsilon[..., m[0]:m[1]])
        Omega_log_pdf_M = self._Omega_log_pdf(self.Ms, m, G, Phi, Upsilon)
        Omega_log_pdf_m = self._Omega_log_pdf(m, m, G, Phi, Upsilon)
        psi_factor = self._psi_factor(G_m, Phi_mm, G_log_pdf)
        result = result | self._T(result['V'], **self._W(**self._A(self._mu_phi_mu(G_log_pdf, Upsilon_log_pdf, Omega_log_pdf_M, Omega_log_pdf_m),
                                                                   self._mu_psi_mu(psi_factor))))
        # result = result | self._A(self._mu_phi_mu(G_log_pdf, Upsilon_log_pdf, Omega_log_pdf_M, Omega_log_pdf_m), self._mu_psi_mu(psi_factor))
        return result

    def _calculate(self):
        """ Called by constructor to calculate all available quantities prior to marginalization.
        These quantities suffice to calculate V[0], V[M], A[00], self.A[m0]=A[M0] and self.A[mm]=A[MM]
        """
        super()._calculate()
        self.Upsilon = self.Lambda2[-1][2]
        self.V2MM = self.V['M'] * self.V['M']
        self.mu_phi_mu = {'pre-factor': tf.reshape(tf.sqrt(Gaussian.det(self.Lambda2[1][0] * self.Lambda2[-1][2])) * self.F, [-1]) if self.is_F_diagonal
                          else tf.sqrt(Gaussian.det(self.Lambda2[1][0] * self.Lambda2[-1][2])) * self.F}
        self.mu_phi_mu['pre-factor'] = tf.reshape(self.mu_phi_mu['pre-factor'], [-1]) if self.is_F_diagonal else self.mu_phi_mu['pre-factor']
        self.G_log_pdf = Gaussian.log_pdf(mean=self.G, variance_cho=tf.sqrt(self.Phi), is_variance_diagonal=True, LBunch=2)
        self.Upsilon_log_pdf = self._Upsilon_log_pdf(self.G, self.Phi, self.Upsilon)
        self.Omega_log_pdf = self._Omega_log_pdf(self.Ms, self.Ms, self.G, self.Phi, self.Upsilon)
        factor = tf.einsum('l, iIN -> liIN', self.g0KY_sum, self.g0)
        self.psi_factor = {'0': tf.squeeze(tf.linalg.triangular_solve(self.K_cho, self._psi_factor_reshape(factor)), axis=-1)}
        self.psi_factor['M'] = self._psi_factor(self.G, self.Phi, self.G_log_pdf)
        A_mislabelled = self._A(self._mu_phi_mu(self.G_log_pdf, self.Upsilon_log_pdf, self.Omega_log_pdf, self.Omega_log_pdf, is_constructor=True),
                                self._mu_psi_mu(self.psi_factor['M'], is_constructor=True))
        self.A = {'00': A_mislabelled.pop('Mm')}
        if not self.options['is_T_partial']:
            self.W = self._W(**A_mislabelled)
            self.A = self.A | A_mislabelled


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
