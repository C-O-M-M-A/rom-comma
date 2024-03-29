#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2024 Robert A. Milton. All rights reserved.
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

""" Contains the calculation of a single closed Sobol index without storing it."""

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.gpr.models import GPR
from romcomma.gsa.base import Calibrator, Gaussian, diag_det


class ClosedSobol(gf.Module, Calibrator):
    """ Calculates closed Sobol Indices."""

    def _serialize_to_tensors(self):
        raise NotImplementedError()

    def _restore_from_tensors(self, restored_tensors):
        raise NotImplementedError()

    @classmethod
    @property
    def META(cls) -> Dict[str, Any]:
        """ Default calculation meta.

        Returns: An empty dictionary.
        """
        return {}

    def marginalize(self, m: TF.Slice) -> Dict[str, TF.Tensor]:
        """ Calculate everything.
        Args:
            m: A Tf.Tensor pair of ints indicating the slice [m[0]:m[1]].
        Returns: The Sobol ClosedSobol of m.
        """
        G, Phi = self.G[..., m[0]:m[1]], self.Phi[..., m[0]:m[1]]
        result = {'V': self._V(G, Phi)}
        result['S'] = result['V'] / self.V[2]
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
        PhiGauss = Gaussian(mean=G, variance=Phi, is_variance_diagonal=True, LBunch=2)
        H = Gaussian(mean=PhiG, variance=PsiPhi, ordinate=G[..., tf.newaxis, tf.newaxis, tf.newaxis, :], is_variance_diagonal=True, LBunch=2)
        H /= PhiGauss.expand_dims([-1, -2, -3])   # Symmetric in L^4 N^2
        # print(sym_check(H, [0, 1, 2, 4, 3, 5])) note the symmetry.
        V = tf.einsum('lLN, lLNjJn, jJn -> lj', self.g0KY, H.pdf, self.g0KY)    # Symmetric in L^2
        return V

    def _calibrate(self):
        """ Called by constructor to calculate all available quantities prior to marginalization.
        These quantities suffice to calculate V[0], V[M].
        """
        pre_factor = tf.sqrt(diag_det(self.Lambda2[1][0] * self.Lambda2[-1][1])) * self.F
        self.g0 = tf.exp(Gaussian(mean=self.gp.X[tf.newaxis, tf.newaxis, ...], variance=self.Lambda2[1][1], is_variance_diagonal=True, LBunch=2).exponent)
        self.g0 *= pre_factor[..., tf.newaxis]     # Symmetric in L^2
        self.g0KY = self.g0 * self.K_inv_Y     # NOT symmetric in L^2
        self.g0KY -= tf.einsum('lLN -> l', self.g0KY)[..., tf.newaxis, tf.newaxis]/tf.cast(tf.reduce_prod(self.g0KY.shape[1:]), dtype=FLOAT())
        self.G = tf.einsum('lLM, NM -> lLNM', self.Lambda2[-1][1], self.gp.X)     # Symmetric in L^2
        self.Phi = self.Lambda2[-1][1]     # Symmetric in L^2
        self.V = {0: self._V(self.G, self.Phi)}     # Symmetric in L^2
        self.V |= {1: tf.linalg.diag_part(self.V[0])}
        V = tf.sqrt(self.V[1])
        self.V |= {2: tf.einsum('l, i -> li', V, V)}
        self.S = self.V[0]/self.V[2]

    def _Lambda2(self) -> Dict[int, Tuple[TF.Tensor]]:
        """ Calculate and cache the required powers of <Lambda^2 + J>.

        Returns: {1: <Lambda^2 + J>, -1: <Lambda^2 + J>^(-1)} for J in {0,1,2}.
        """
        if self.is_F_diagonal:
            result = tf.einsum('lM, lM -> lM', self.Lambda, self.Lambda)[:, tf.newaxis, :]
        else:
            result = tf.einsum('lM, LM -> lLM', self.Lambda, self.Lambda)
        result = tuple(result + j for j in range(3))
        return {1: result, -1: tuple(value**(-1) for value in result)}

    def __init__(self, gp: GPR, **kwargs: Any):
        """ Construct a ClosedSobol object. A wide range of values are collected or calculated and cached, especially via the final call to self._calibrate.

        Args:
            gp: The gp to analyze.
            **kwargs: The calculation meta to override META.
        """
        super().__init__()
        self.gp = gp
        self.meta = self.META | kwargs
        # Unwrap data
        self.L, self.M, self.N = self.gp.L, self.gp.M, self.gp.N
        self.Ms = tf.constant([0, self.M], dtype=INT())
        self.F = tf.constant(self.gp.kernel.data.frames.variance.tf, dtype=FLOAT())
        # Cache the training data kernel
        self.K_cho = tf.constant(self.gp.K_cho, dtype=FLOAT())
        self.K_inv_Y = tf.constant(self.gp.K_inv_Y, dtype=FLOAT())
        # Determine if F is diagonal
        self.is_F_diagonal = self.meta.pop('is_F_diagonal', None)
        if self.is_F_diagonal is None:
            gp_options = self.gp.read_meta() if self.gp._meta_json.exists() else self.gp.META
            self.is_F_diagonal = not gp_options.pop('kernel', {}).pop("covariance", False)
        # Reshape according to is_F_diagonal
        if self.is_F_diagonal:
            self.F = self.F if self.F.shape[0] == 1 else tf.linalg.diag_part(self.F)
            self.F = tf.reshape(self.F, [self.L, 1])
        else:
            self.K_inv_Y = tf.transpose(self.K_inv_Y, [1, 0, 2])
        # Set Lambdas
        self.Lambda = tf.broadcast_to(tf.constant(self.gp.kernel.data.frames.lengthscales.np, dtype=FLOAT()), [self.L, self.M])
        self.Lambda2 = self._Lambda2()
        # Calculate and store values for m=0 and m=M
        self._calibrate()


class ClosedSobolWithError(ClosedSobol):
    """ Calculates closed Sobol Indices with Errors."""

    @classmethod
    @property
    def META(cls) -> Dict[str, Any]:
        """ Default calculation meta. ``is_T_partial`` forces W[Mm] = W[MM] = 0.

        Returns:
            is_T_partial: If True this effectively asserts the full ['M'] model is variance free, so WmM is not calculated or returned.
        """
        return {'is_T_partial': True}

    class RankEquation(NamedTuple):
        l: str
        i: str
        j: str
        k: str

    class RankEquations(NamedTuple):
        DIAGONAL: Any
        MIXED: Any

    RANK_EQUATIONS: RankEquations = RankEquations(DIAGONAL=(RankEquation(l='j', i='k', j='l', k='i'), RankEquation(l='k', i='j', j='i', k='l')),
                                                  MIXED=(RankEquation(l='k', i='k', j='j', k='i'),))

    def _equateRanks(self, liLNjkJM: TF.Tensor, rank_eq: RankEquation) -> TF.Tensor:
        """ Equate the ranks of a tensor, according to eqRanks.

        Args:
            liLNjkJM: A tensor which must have ranks liLNjkJM.
            rank_eq: Which ranks to equate.

        Returns:
            LNjkS or LNjiS.
        """
        shape = liLNjkJM.shape.as_list()
        eqRanks_j = 'j' if shape[4] == 1 else rank_eq.j
        eqRanks_k = 'k' if shape[5] == 1 else rank_eq.k
        liLNjkJM = tf.reshape(liLNjkJM, shape[:-2] + [-1])  # TensorFlow only does einsum up to rank 6!
        if rank_eq in self.RANK_EQUATIONS.MIXED:
            result = tf.einsum(f'iiLNjkS -> LNjiS', liLNjkJM)
        else:
            result = tf.einsum(f'liLN{eqRanks_j}{eqRanks_k}S -> LN{rank_eq.j}{rank_eq.k}S', liLNjkJM)
        result = tf.reshape(result, result.shape[:-1].as_list() + shape[-2:])  # TensorFlow only does einsum up to rank 6!
        return tf.einsum(f'LNjjJM -> LNjJM', result)[..., tf.newaxis, :, :] if rank_eq.j == 'i' else result

    def _equatedRanksGaussian(self, mean: TF.Tensor, variance: TF.Tensor, ordinate: TF.Tensor, rank_eqs: Tuple[RankEquation]) -> List[Gaussian]:
        """ Equate ranks and calculate Gaussian.

        Args:
            mean: liLNjkJn.
            variance: liLjkJM.
            ordinate: liLNM and jkJnM.
            rank_eqs: A tuple of RankEquators to apply.
        Returns: liLNjkJn.

        """
        result = []
        N_axis = 3
        for rank_eq in rank_eqs:
            eq_ranks_variance = self._equateRanks(tf.expand_dims(variance, N_axis), rank_eq)[..., tf.newaxis, :]
            eq_ranks_mean = self._equateRanks(mean, rank_eq)[..., tf.newaxis, :]
            shape = tf.concat([eq_ranks_mean.shape[:-2], ordinate.shape[-2:]], axis=0) if tf.rank(ordinate) > 2 else None
            eq_ranks_mean = (eq_ranks_mean if shape is None else tf.broadcast_to(eq_ranks_mean, shape)) - ordinate
            result += [Gaussian(mean=eq_ranks_mean, variance=eq_ranks_variance, is_variance_diagonal=True, LBunch=10000)]
        return result

    def _OmegaGaussian(self, mp: TF.Slice, G: TF.Tensor, Phi: TF.Tensor, Upsilon: TF.Tensor, rank_eqs: Tuple[RankEquation]) -> List[Gaussian]:
        """ The Omega integral for m=mp or m=mp=[:M]. Does not apply when m=[0:0].

        Args:
            mp: The marginalization m_primed.
            G: Un-marginalized. lLNM and jJnM.
            Phi: Un-marginalized. ikM and jJM.
            Upsilon: Un-marginalized. ikM.
            rank_eqs: A tuple of RankEquators to apply.
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
        return self._equatedRanksGaussian(mean, variance, G[:, tf.newaxis, ...], rank_eqs)

    def _UpsilonGaussian(self, G: TF.Tensor, Phi: TF.Tensor, Upsilon: TF.Tensor, rank_eqs: Tuple[RankEquation]) -> List[Gaussian]:
        """ The Upsilon integral.

        Args:
            G: lLNM.
            Phi: lLM.
            Upsilon: ikM.
            rank_eqs: A tuple of RankEquators to apply.
        Returns: liLNjkJn.
        """
        Upsilon_cho = tf.sqrt(Upsilon)
        mean = tf.einsum('ikM, lLNM -> liLNkM', Upsilon_cho, G)[..., tf.newaxis, :, tf.newaxis, :]
        variance = 1 - tf.einsum('ikM, lLM, ikM -> liLkM', Upsilon_cho, Phi, Upsilon_cho)[..., tf.newaxis, :, tf.newaxis, :]
        return self._equatedRanksGaussian(mean, variance, tf.constant(0, dtype=FLOAT()), rank_eqs)

    def _mu_phi_mu(self, GGaussian: Gaussian, UpsilonGaussians: List[Gaussian], OmegaGaussians: List[Gaussian], rank_eqs: Tuple[RankEquation]) -> TF.Tensor:
        """ Calculate E_m E_mp (mu[m] phi[m][mp] mu[mp]).

        Args:
            GGaussian: jJn.
            UpsilonGaussians: liLNjk.
            OmegaGaussians: liLNjkJn.
            rank_eqs: A tuple of RankEquators to apply.
        Returns: li.
        """
        GGaussian = GGaussian.expand_dims([2])
        mu_phi_mu = 0.0
        for i, rank_eq in enumerate(rank_eqs):
            OmegaGaussians[i] /= GGaussian
            OmegaGaussians[i].exponent += UpsilonGaussians[i].exponent
            if UpsilonGaussians[i].cho_diag.shape[-1] == GGaussian.cho_diag.shape[-1]:
                OmegaGaussians[i].cho_diag *= UpsilonGaussians[i].cho_diag
            else:
                OmegaGaussians[i].cho_diag = (diag_det(OmegaGaussians[i].cho_diag) * diag_det(UpsilonGaussians[i].cho_diag))[..., tf.newaxis]
            if rank_eq in self.RANK_EQUATIONS.MIXED:
                result = tf.einsum('kLN, LNjkJn, jJn -> jk', self.g0KY, OmegaGaussians[i].pdf, self.g0KY)
                mu_phi_mu += tf.einsum('k, jk -> jk', self.mu_phi_mu['pre-factor'], result)
                mu_phi_mu = tf.linalg.set_diag(mu_phi_mu, 2 * tf.linalg.diag_part(mu_phi_mu))
            elif rank_eq.l == 'k' and rank_eq.i == 'j':
                result = tf.einsum('jLN, LNjkJn, jJn -> j', self.g0KY, OmegaGaussians[i].pdf, self.g0KY)
                mu_phi_mu += tf.linalg.diag(tf.einsum('j, j -> j', self.mu_phi_mu['pre-factor'], result))
            else:
                result = tf.einsum(f'jLN, LNjkJn, jJn -> jk', self.g0KY, OmegaGaussians[i].pdf, self.g0KY)
                mu_phi_mu += tf.einsum(f'k, jk -> jk', self.mu_phi_mu['pre-factor'], result)
        return mu_phi_mu

    def _psi_factor(self, G: TF.Tensor, Phi: TF.Tensor, GGaussian: Gaussian) -> TF.Tensor:
        """ Calculate the psi_factor E_m or E_mp for E_m E_mp (mu[m] psi[m][mp] mu[mp])

        Args:
            G: lLNm
            Phi: lLm
            GGaussian: lLn
        Returns: liS
        """
        D = Phi[..., tf.newaxis, tf.newaxis, :] - tf.einsum('lLM, iIM, lLM -> lLiIM', Phi, Phi, Phi)
        mean = tf.einsum('lLM, iInM -> lLiInM', Phi, G)
        mean = mean[:, :, tf.newaxis, ...] - G[..., tf.newaxis, tf.newaxis, tf.newaxis, :]
        gaussian = Gaussian(mean=mean, variance=D, is_variance_diagonal=True, LBunch=2)
        gaussian /= GGaussian.expand_dims([-1, -2, -3])
        factor = tf.einsum('lLN, iIn, lLNiIn -> liIn', self.g0KY, self.g0, gaussian.pdf)
        if tf.rank(self.K_cho) == 2 and factor.shape[-2] == 1:
            factor = tf.einsum('lNiI -> liIN', tf.linalg.diag(tf.einsum('liIN -> lNi', factor)))
        factor = tf.reshape(factor, factor.shape[:-2].as_list() + [-1, 1])
        factor = tf.squeeze(tf.linalg.triangular_solve(self.K_cho, factor), axis=-1)
        return factor

    def _mu_psi_mu(self, psi_factor: TF.Tensor, rank_eqs: Tuple[RankEquation]) -> TF.Tensor:
        """ Multiply psi_factors to calculate mu_psi_mu.

        Args:
            psi_factor: liS.
            rank_eqs: A tuple of RankEquators to apply.
        Returns: li
        """
        first_psi_factor = self.psi_factor if rank_eqs is self.RANK_EQUATIONS.MIXED else psi_factor
        first_ein = 'liS' if rank_eqs is self.RANK_EQUATIONS.DIAGONAL else 'iiS'
        result = tf.einsum(f'{first_ein}, liS -> li', first_psi_factor, psi_factor)
        return tf.linalg.set_diag(result, 2 * tf.linalg.diag_part(result))

    def _W(self, mu_phi_mu: TF.Tensor, mu_psi_mu: TF.Tensor) -> TF.Tensor:
        """ Calculate W.

        Returns: W[mm] if is_T_partial, else W{mm, Mm}
        """
        W = mu_phi_mu - mu_psi_mu
        W += tf.transpose(W)
        return W

    def _T(self, Wmm: TF.Tensor, WMm: TF.Tensor = None, Vm: TF.Tensor = None) -> TF.Tensor:
        """ Calculate T

        Args:
            Wmm: li
            WMm: li
            Vm: li
        Returns: The closed index uncertainty T.
        """
        if self.meta['is_T_partial']:
            Q = Wmm
        else:
            Q = Wmm - 2 * Vm * WMm / self.V[1] + Vm * Vm * self.Q
        return tf.sqrt(tf.abs(Q) / self.V[4])

    def marginalize(self, m: TF.Slice) -> Dict[str: TF.Tensor]:
        """ Calculate everything.
        Args:
            m: A Tf.Tensor pair of ints indicating the slice [m[0]:m[1]].
        Returns: The Sobol ClosedSobol of m, with errors (T and W).
        """
        result = super().marginalize(m)
        G, Phi, Upsilon = tuple(tensor[..., m[0]:m[1]] for tensor in (self.G, self.Phi, self.Upsilon))
        GGaussian = Gaussian(G, Phi, is_variance_diagonal=True, LBunch=2)
        psi_factor = self._psi_factor(G, Phi, GGaussian)
        if self.meta['is_T_partial']:
            UpsilonGaussians = self._UpsilonGaussian(G, Phi, Upsilon, self.RANK_EQUATIONS.DIAGONAL)
            OmegaGaussians = self._OmegaGaussian(m, self.G, self.Phi, self.Upsilon, self.RANK_EQUATIONS.DIAGONAL)
            Wmm = self._W(self._mu_phi_mu(GGaussian, UpsilonGaussians, OmegaGaussians, self.RANK_EQUATIONS.DIAGONAL),
                          self._mu_psi_mu(psi_factor, self.RANK_EQUATIONS.DIAGONAL))
            result |= {'W': Wmm, 'T': self._T(Wmm)}
        else:
            UpsilonGaussians = self.RankEquations(*(self._UpsilonGaussian(G, Phi, Upsilon, rank_eqs) for i, rank_eqs in enumerate(self.RANK_EQUATIONS)))
            OmegaGaussians = self.RankEquations(*(self._OmegaGaussian(m, self.G, self.Phi, self.Upsilon, rank_eqs)
                                                 for i, rank_eqs in enumerate(self.RANK_EQUATIONS)))
            Wmm = (self._W(self._mu_phi_mu(GGaussian, UpsilonGaussians.DIAGONAL, OmegaGaussians.DIAGONAL, self.RANK_EQUATIONS.DIAGONAL),
                           self._mu_psi_mu(psi_factor, self.RANK_EQUATIONS.DIAGONAL)))
            WMm = self._W(self._mu_phi_mu(GGaussian, self.UpsilonGaussians.MIXED, OmegaGaussians.MIXED, self.RANK_EQUATIONS.MIXED),
                          self._mu_psi_mu(psi_factor, self.RANK_EQUATIONS.MIXED))
            result |= {'W': Wmm, 'T': self._T(Wmm, WMm, result['V'])}
        return result

    def _calibrate(self):
        """ Called by constructor to calculate all available quantities prior to marginalization.
        These quantities suffice to calculate V[0], V[M], A[00], self.A[m0]=A[M0] and self.A[mm]=A[MM]
        """
        super()._calibrate()
        if not self.is_F_diagonal:
            raise NotImplementedError('If the MOGP kernel covariance is not diagonal, the Sobol error calculation is unstable.')
        self.Upsilon = self.Lambda2[-1][2]
        self.V |= {4: tf.einsum('li, li -> li', self.V[2], self.V[2])}
        self.mu_phi_mu = {'pre-factor': tf.reshape(tf.sqrt(tf.reduce_prod(self.Lambda2[1][0] * self.Lambda2[-1][2], axis=-1)) * self.F, [-1])}
        self.mu_phi_mu['pre-factor'] = tf.reshape(self.mu_phi_mu['pre-factor'], [-1])
        self.GGaussian = Gaussian(mean=self.G, variance=self.Phi, is_variance_diagonal=True, LBunch=2)
        self.psi_factor = self._psi_factor(self.G, self.Phi, self.GGaussian)
        if self.meta['is_T_partial']:
            self.UpsilonGaussians = self._UpsilonGaussian(self.G, self.Phi, self.Upsilon, self.RANK_EQUATIONS.DIAGONAL)
            self.OmegaGaussians = self._OmegaGaussian(self.Ms, self.G, self.Phi, self.Upsilon, self.RANK_EQUATIONS.DIAGONAL)
            self.W = self._W(self._mu_phi_mu(self.GGaussian, self.UpsilonGaussians, self.OmegaGaussians, self.RANK_EQUATIONS.DIAGONAL),
                             self._mu_psi_mu(self.psi_factor, self.RANK_EQUATIONS.DIAGONAL))
        else:
            self.UpsilonGaussians = self.RankEquations(*(self._UpsilonGaussian(self.G, self.Phi, self.Upsilon, rank_eq)
                                                         for i, rank_eq in enumerate(self.RANK_EQUATIONS)))
            self.OmegaGaussians = self.RankEquations(*(self._OmegaGaussian(self.Ms, self.G, self.Phi, self.Upsilon, rank_eq)
                                                       for i, rank_eq in enumerate(self.RANK_EQUATIONS)))
            self.W = self.RankEquations(*(self._W(self._mu_phi_mu(self.GGaussian, self.UpsilonGaussians[i], self.OmegaGaussians[i], rank_eq),
                                                  self._mu_psi_mu(self.psi_factor, rank_eq)) for i, rank_eq in enumerate(self.RANK_EQUATIONS)))
            self.Q = tf.linalg.diag_part(self.W.MIXED) / (4.0 * self.V[1] * self.V[1])
            self.Q = self.Q[tf.newaxis, ...] + self.Q[..., tf.newaxis] + 2.0 * tf.linalg.diag(self.Q)
            self.T = self._T(self.W.DIAGONAL, self.W.MIXED, self.V[0])


class ClosedSobolWithRotation(ClosedSobol):
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
