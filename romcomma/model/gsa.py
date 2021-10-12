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

""" Contains Global Sensitivity Analysis tools."""

from __future__ import annotations

from romcomma.typing_ import *


# noinspection PyPep8Naming
class Sobol(Model):
    """ Interface to a Sobol' Index Calculator and Optimizer.

    Internal quantities are called variant if they depend on Theta, invariant otherwise.
    Invariants are calculated in the constructor. Variants are calculated in Theta.setter."""

    """ Required overrides."""

    MEMORY_LAYOUT = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-All-input-is-Fortran-layout)."

    Parameters = NamedTuple("Parameters", [('Theta', NP.Matrix), ('D', NP.Matrix), ('S1', NP.Matrix), ('S', NP.Matrix), ('ST', NP.Matrix)])
    """ 
        **Theta** -- The (Mu, M) rotation matrix ``U = X Theta.T`` (in design matrix terms), so u = Theta x (in column vector terms).

        **D** -- An (L L, M) Matrix of cumulative conditional variances D[l,k,m] = S[l,k,m] D[l,k,M].

        **S1** -- An (L L, M) Matrix of Sobol' first order indices.

        **S** -- An (L L, M) Matrix of Sobol' closed indices.

        **ST** -- An (L L, M) Matrix of Sobol' total indices.
    """
    DEFAULT_PARAMETERS = Parameters(*(atleast_2d(None),) * 5)

    # noinspection PyPep8Naming
    class SemiNorm:
        """Defines a SemiNorm on (L,L) matrices, for use by Sobol.

        Attributes:
            value: The SemiNorm.value function, which is Callable[[Tensor], ArrayLike] so it is vectorizable.
            derivative: The SemiNorm.derivative function, which is Callable[[Matrix], Matrix] as it is not vectorizable.
        """

        DEFAULT_META = {'classmethod': 'element', 'L': 1, 'kwargs': {'row': 0, 'column': 0}}

        @classmethod
        def from_meta(cls, meta: Union[Dict, 'Sobol.SemiNorm']) -> Sobol.SemiNorm:
            """ Create a SemiNorm from meta information. New SemiNorms should be registered in this function.

            Args:
                meta: A Dict containing the meta data for function construction. Use SemiNorm.DEFAULT_META as a template.
                    Otherwise, meta in the form of a SemiNorm is just returned verbatim, anything elses raises a TypeError.
            Returns: The SemiNorm constructed according to meta.

            Raises:
                TypeError: Unless meta must is a Dict or a SemiNorm.
                NotImplementedError: if meta['classmethod'] is not recognized by this function.
            """
            if isinstance(meta, Sobol.SemiNorm):
                return meta
            if not isinstance(meta, dict):
                raise TypeError("SemiNorm meta data must be a Dict or a SemiNorm, not a {0}.".format(type(meta)))
            if meta['classmethod'] == 'element':
                return cls.element(meta['L'], **meta['kwargs'])
            else:
                raise NotImplementedError("Unrecognized meta['classmethod'] = '{0}'. ".format(meta['classmethod']) +
                                          "Please implement the relevant @classmethod in Sobol.SemiNorm " +
                                          "and register it in Sobol.SemiNorm.from_meta().")

        @classmethod
        def element(cls, L: int, row: int, column: int) -> Sobol.SemiNorm:
            """ Defines a SemiNorm on (L,L) matrices which is just the (row, column) element.
            Args:
                L:
                row:
                column:
            Returns: A SemiNorm object encapsulating the (row, column) element semi-norm on (L,L) matrices.

            Raises:
                ValueError: If row or column not in range(L).
            """
            if not 0 <= row <= L:
                raise ValueError("row {0:d} is not in range(L={1:d}.".format(row, L))
            if not 0 <= column <= L:
                raise ValueError("column {0:d} is not in range(L={1:d}.".format(column, L))
            meta = {'classmethod': 'element', 'L': L, 'kwargs': {'row': row, 'column': column}}
            _derivative = zeros((L, L), dtype=float)
            _derivative[row, column] = 1.0

            def value(D: NP.Tensor) -> NP.ArrayLike:
                return D[row, column]

            def derivative(D: NP.Matrix) -> NP.Matrix:
                return _derivative

            return Sobol.SemiNorm(value, derivative, meta)

        def __init__(self, value: Callable[[NP.Tensor], NP.ArrayLike], derivative: Callable[[NP.Matrix], NP.Matrix], meta: Dict):
            """ Construct a SemiNorm on (L,L) matrices.

            Args:
                value: A function mapping an (L,L) matrix D to a float SemiNorm.value
                derivative: A function mapping an (L,L) matrix D to an (L,L) matrix SemiNorm.derivative = d SemiNorm.value / (d D).
                meta: A Dict similar to SemiNorm.DEFAULT_META, giving precise information to construct this SemiNorm
            """
            self.value = value
            self.derivative = derivative
            self.meta = meta

    DEFAULT_OPTIMIZER_OPTIONS = {'semi_norm': SemiNorm.DEFAULT_META, 'N_exploit': 0, 'N_explore': 0, 'options': {'gtol': 1.0E-12}}
    """ 
        **semi_norm** -- A Sobol.SemiNorm on (L,L) matrices defining the Sobol' measure to optimize against.

        **N_exploit** -- The number of exploratory xi vectors to exploit (gradient descend) in search of the global optimum.
            If N_exploit < 1, only re-ordering of the input basis is allowed.

        **N_explore** -- The number of random_sgn xi vectors to explore in search of the global optimum. 
            If N_explore <= 1, gradient descent is initialized from Theta = Identity Matrix.

        **options** -- A Dict of options passed directly to the underlying optimizer.
    """

    NAME = "sobol"

    """ End of required overrides."""

    @property
    def gp(self):
        """ The underlying GP."""
        return self._gp

    @property
    def D(self) -> NP.Tensor3:
        """ An (L, L, Mx) Tensor3 of conditional variances."""
        return self._D

    @property
    def S(self) -> NP.Tensor3:
        """ An (L, L, M) Tensor3 of Closed (cumulative) Sobol' indices."""
        return self._D / self._D[:, :, -1]

    @property
    def S1(self) -> NP.Tensor3:
        """ An (L, L, Mx) Tensor3 of first order (main effect) Sobol' indices."""
        return self._S1

    @property
    def ST(self) -> NP.Tensor3:
        """ An (L, L, Mx) Tensor3 of Total Sobol' indices."""
        return self._ST

    @property
    def lengthscales(self):
        """ An (Mx,) Array of RBF lengthscales."""
        return self._lengthscale

    def Tensor3AsMatrix(self, DorS: NP.Tensor3) -> NP.Matrix:
        return reshape(DorS, (self._L * self._L, self._M))

    # @property
    # def Theta_old(self) -> NP.Matrix:
    #     """ Sets or gets the (M, M) rotation Matrix, prior to updates by xi. Setting automatically updates Theta, triggering Sobol' recalculation."""
    #     return self._Theta_old
    #
    # @Theta_old.setter
    # def Theta_old(self, value: NP.Matrix):
    #     assert value.shape == (self._M, self._M)
    #     self._Theta_old = value
    #     self.Theta = self.Theta_old[:self._m + 1, :].copy(order=self.MEMORY_LAYOUT)
    #
    # @property
    # def Theta(self) -> NP.Matrix:
    #     """ Sets or gets the (m+1, Mx) rotation Matrix which has been updated by xi. Setting triggers Sobol' recalculation."""
    #     return self._Theta
    #
    # # noinspection PyAttributeOutsideInit
    # @Theta.setter
    # def Theta(self, value: NP.Matrix):
    #     """ Complete calculation of variants and Sobol' indices (conditional variances _D actually) is found here and here only."""
    #
    #     assert value.shape == (self._m + 1, self._M)
    #     self._Theta = value
    #
    #     """ Calculate variants related to Sigma. """
    #     self._Sigma_partial = einsum('M, kM -> Mk', self._Sigma_diagonal, self.Theta, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     Sigma = einsum('mM, Mk -> mk', self.Theta, self._Sigma_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self._Sigma_cho = cho_factor(Sigma, lower=False, overwrite_a=False, check_finite=False)
    #     Sigma_cho_det = prod(diag(self._Sigma_cho[0]))
    #     self._2I_minus_Sigma_partial = einsum('M, kM -> Mk', 2 - self._Sigma_diagonal, self.Theta, optimize=True, dtype=float,
    #                                           order=self.MEMORY_LAYOUT)
    #     _2I_minus_Sigma = einsum('mM, Mk -> mk', self.Theta, self._2I_minus_Sigma_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self._2I_minus_Sigma_cho = cho_factor(_2I_minus_Sigma, lower=False, overwrite_a=False, check_finite=False)
    #     _2I_minus_Sigma_cho_det = prod(diag(self._2I_minus_Sigma_cho[0]))
    #     self._inv_Sigma_Theta = cho_solve(self._Sigma_cho, self.Theta, overwrite_b=False, check_finite=False)
    #     T_inv_Sigma_T = atleast_2d(einsum('NM, mM, mK, NK -> N', self._FBold, self.Theta, self._inv_Sigma_Theta, self._FBold,
    #                                       optimize=True, dtype=float, order=self.MEMORY_LAYOUT))
    #     self._D_const = (_2I_minus_Sigma_cho_det * Sigma_cho_det) ** (-1)
    #
    #     """ Calculate variants related to Phi. """
    #     self._Phi_partial = einsum('M, kM -> Mk', self._Psi_diagonal, self.Theta, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     Phi = einsum('mM, Mk -> mk', self.Theta, self._Phi_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self._Phi_cho = cho_factor(Phi, lower=False, overwrite_a=False, check_finite=False)
    #     self._inv_Phi_inv_Sigma_Theta = cho_solve(self._Phi_cho, self._inv_Sigma_Theta, overwrite_b=False, check_finite=False)
    #     T_inv_Phi_inv_Sigma_T = einsum('NOM, mM, mK, NOK -> NO', self._V_pre_outer_square, self.Theta, self._inv_Phi_inv_Sigma_Theta,
    #                                     self._V_pre_outer_square, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #
    #     """ Finally calculate conditional variances _D."""
    #     self._W = exp(-0.5 * (T_inv_Sigma_T + transpose(T_inv_Sigma_T) - T_inv_Phi_inv_Sigma_T))
    #     self._D_plus_Ft_1_Ft = self._D_const * einsum('LN, NO, KO -> LK', self._fBold_bar_0, self._W, self._fBold_bar_0,
    #                                                   optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self._D[:, :, self._m] = self._D_plus_Ft_1_Ft - self._f_bar_0_2
    #
    # @property
    # def xi(self) -> NP.Array:
    #     """ Sets or gets the (Mx-m-1) Array which is the row m update to Theta. Setting updates Theta, so triggers Sobol' recalculation."""
    #     return self._xi
    #
    # @xi.setter
    # def xi(self, value: NP.Array):
    #     assert value.shape[0] == self._xi_len
    #     norm = 1 - einsum('m, m -> ', value, value, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     if norm < EFFECTIVELY_ZERO:
    #         value *= sqrt((1-EFFECTIVELY_ZERO)/(1-norm))
    #         norm = EFFECTIVELY_ZERO
    #     self._xi = append(sqrt(norm), value)
    #     self.Theta[self._m, :] = einsum('k, kM -> M', self._xi, self.Theta_old[self._m:, :],
    #                                    optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self.Theta = self.Theta[:self._m + 1, :]

    def optimize(self, **kwargs):
        """ Optimize ``Theta`` to maximize ``semi_norm(D[:,:,m])`` for ``m=0,1,..(Mu-1)``.

        Args:
            options: A Dict similar to (and documented in) Sobol.DEFAULT_OPTIMIZER_OPTIONS.
        Raises:
            TypeError: Unless options['SemiNorm'] is a Sobol.SemiNorm or a Dict.
            UserWarning: If L for the SemiNorm must be changed to match self._L.
        """
        options = deepcopy(kwargs)
        if options is None:
            options = self._read_options() if self._options_json.exists() else self.DEFAULT_OPTIMIZER_OPTIONS
        semi_norm = Sobol.SemiNorm.from_meta(options['semi_norm'])
        if semi_norm.meta['L'] != self._L:
            warn("I am changing Sobol.semi_norm.meta['L'] from {0:d} to {1:d} = Sobol.gp.L.".format(semi_norm.meta['L'], self._L))
            semi_norm.meta['L'] = self._L
            semi_norm = Sobol.SemiNorm.from_meta(semi_norm.meta)
        options['semi_norm'] = semi_norm.meta
        self._write_options(options)
        #
        # def _objective_value(xi: NP.Array) -> float:
        #     """ Maps ``xi`` to the optimization objective (a conditional variance ``D'').
        #
        #     Args:
        #         xi: The Theta row update.
        #     Returns: The scalar (float) -semi_norm(D[:, :, m]).
        #     """
        #     self.xi = xi
        #     return -semi_norm.value(self._D[:, :, self._m])
        #
        # def _objective_jacobian(xi: NP.Array) -> NP.Array:
        #     """ Maps ``xi`` to an ``(Mx-m,) Array``, the optimization objective jacobian.
        #
        #     Args:
        #         xi: The Theta row update.
        #     Returns: The (Mx-m,) jacobian Array -d(semi_norm(D[:, :, m])) / (d xi).
        #     """
        #     self.xi = xi
        #     return -einsum('LK, LKj -> j', semi_norm.derivative(self.D[:, :, self._m]), self.D_jacobian,
        #                    optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
        #
        # self._objective_value, self._objective_jacobian = _objective_value, _objective_jacobian
        #
        # if options['N_exploit'] >= 1:
        #     q = None
        #     for self.m in range(self._Mu):
        #         if q is not None:
        #             self.Theta_old = transpose(q)
        #         self.xi = self._optimal_rotating_xi(options['N_explore'], options['N_exploit'], options['options'])
        #         q, r = qr(transpose(self.Theta), check_finite=False)
        #         sign_correction = sign(diag(r))
        #         q *= concatenate((sign_correction, ones(self._M - len(sign_correction), dtype=int)))
        #     self.Theta_old = transpose(q)
        # q = None
        # for self.m in range(self._Mu):
        #     if q is not None:
        #         self.Theta_old = transpose(q)
        #     self.xi = self._optimal_reordering_xi()
        #     q, r = qr(transpose(self.Theta), check_finite=False)
        #     # sign_correction = sign(diag(r))
        #     # q *= concatenate((sign_correction, ones(self._M - len(sign_correction), dtype=int)))
        # self.Theta_old = transpose(q)
        # self.write_parameters(self.Parameters(Mu=self.Mu, Theta=self._Theta_old, D=self.Tensor3AsMatrix(self._D), S1=None,
        #                                       S=self.Tensor3AsMatrix(self.S)))
        # self.replace_X_with_U()

    # def write_parameters(self, parameters: Parameters) -> Parameters:
    #     """ Calculate the main Sobol' indices _S1, then write model.parameters to their csv files.
    #
    #     Args:
    #         parameters: The NamedTuple to be the new value for self.parameters.
    #     Returns: The NamedTuple written to csv. Essentially self.parameters, but with Frames in place of Matrices.
    #     """
    #     if self._m is not None:
    #         m_saved = self._m
    #         self.m = 0
    #         xi_temp = zeros(self._xi_len, dtype=float, order=self.MEMORY_LAYOUT)
    #         for m in reversed(range(len(xi_temp))):
    #             xi_temp[m] = 1.0
    #             self.xi = xi_temp
    #             self._S1[:, :, m+1] = self.D[:, :, 0] / self.D[:, :, -1]
    #             xi_temp[m] = 0.0
    #         self.xi = xi_temp
    #         self._S1[:, :, 0] = self.D[:, :, 0] / self.D[:, :, -1]
    #         self.m = m_saved
    #     return super().write_parameters(parameters._replace(S1=self.Tensor3AsMatrix(self._S1)))

    def replace_X_with_U(self):
        """ Replace X with its rotated/reordered version U."""
        column_headings = MultiIndex.from_product(((self._gp.fold.meta['data']['X_heading'],), ("u{:d}".format(i) for i in range(self.Mu))))
        X = DataFrame(einsum('MK, NK -> NM', self.Theta_old, self._gp.X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT),
                      columns=column_headings, index=self._gp.fold.X.index)
        test_X = DataFrame(einsum('MK, NK -> NM', self.Theta_old, self._gp.fold.test_X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT),
                           columns=column_headings, index=self._gp.fold.test_X.index)
        self._gp.fold.data.df = concat((X, self._gp.fold.data.df[[self._gp.fold.meta['data']['Y_heading']]].copy(deep=True)), axis='columns')
        self._gp.fold.data.write()
        self._gp.fold.test.df = concat((test_X, self._gp.fold.test.df[[self._gp.fold.meta['data']['Y_heading']]].copy(deep=True)), axis='columns')
        self._gp.fold.test.write()
        self._gp.fold.meta_data_update()

    def reorder_data_columns(self, reordering: NP.Array):
        """ Reorder the columns in self._gp.data

        Args:
            reordering: An array of column indices specifying the reordering
        """
        columns = self._gp.fold.data.df.columns
        new_columns = columns.levels[1].to_numpy()
        new_columns[:reordering.shape[0]] = new_columns[reordering]
        columns = columns.reindex(new_columns, level=1)
        self._gp.fold.data.df.reindex(columns=columns, copy=False)
        self._gp.fold.data.write()
        self._gp.fold.test.df.reindex(columns=columns, copy=False)
        self._gp.fold.test.write()

    # def exploratory_xi(self, N_explore: int) -> NP.Matrix:
    #     """ Generate a matrix of xi's to explore.
    #
    #     Args:
    #         N_explore: The maximum number of xi's to explore. If N_explore &le 1 the zero vector is returned.
    #             This is tantamount to taking Theta = I (the identity matrix).
    #         xi_len: The length of each xi (row) Array.
    #     Returns: An (N_explore, xi_len) Matrix where 1 &le N_explore &le max(N_explore,1).
    #     """
    #     assert self._xi_len > 0
    #     if N_explore <= 1:
    #         return zeros((1, self._xi_len))
    #     else:
    #         N_explore = round(N_explore**((self._xi_len + 1) / self._M))
    #         dist = distribution.Multivariate.Independent(self._xi_len + 1, distribution.Univariate('uniform', loc=-1, scale=2))
    #         result = dist.sample(N_explore, distribution.SampleDesign.LATIN_HYPERCUBE)
    #         norm = sqrt(sum(result ** 2, axis=1).reshape((N_explore, 1)))
    #         return result[:, 1:] / norm
    #
    #     """
    #     if N_explore <= 1:
    #         return zeros((1, xi_len))
    #     elif N_explore < 3 ** xi_len:
    #         result = random.randint(3, size=(N_explore, xi_len)) - 1
    #     else:
    #         values = array([-1, 0, 1])
    #         # noinspection PyUnusedLocal
    #         values = [values.copy() for i in range(xi_len)]
    #         result = meshgrid(*values)
    #         result = [ravel(arr, order=cls.MEMORY_LAYOUT) for arr in result]
    #         result = concatenate(result, axis=0)
    #     return result * xi_len ** (-1 / 2)
    #     """
    #
    # def _optimal_rotating_xi(self, N_explore: int, N_exploit: int, options: Dict) -> NP.Array:
    #     """ Optimizes the ``Theta`` row update ``xi`` by allowing general rotation.
    #
    #     Args:
    #         N_explore: The number of random_sgn xi vectors to explore in search of the global optimum.
    #         N_exploit: The number of exploratory xi vectors to exploit (gradient descend) in search of the global optimum.
    #         options: A Dict of options passed directly to the underlying optimizer.
    #     Returns:
    #         The Array xi of euclidean length &le 1 which maximizes self.optimization_objective(xi).
    #     """
    #     explore = self.exploratory_xi(N_explore)
    #     best = [[0, explore[0]]] * N_exploit
    #     for xi in explore:
    #         objective_value = self._objective_value(xi)
    #         for i in range(N_exploit):
    #             if objective_value < best[i][0]:
    #                 for j in reversed(range(i + 1, N_exploit)):
    #                     best[j] = best[j - 1]
    #                 best[i] = [objective_value, xi]
    #                 break
    #     for record in best:
    #         result = optimize.minimize(self._objective_value, record[1], method='BFGS', jac=self._objective_jacobian, options=options)
    #         if result.fun < best[0][0]:
    #             best[0] = [result.fun, result.x]
    #     return best[0][1]
    #
    # def _optimal_reordering_xi(self) -> NP.Array:
    #     """ Optimizes the ``Theta`` row update ``xi`` by allowing re-ordering only.
    #
    #     Returns:
    #         The Array xi consisting of all zeros except at most one 1.0 which maximizes self.optimization_objective(xi).
    #     """
    #     xi = zeros(self._xi_len, dtype=float, order=self.MEMORY_LAYOUT)
    #     best = self._objective_value(xi), self._xi_len
    #     for m in range(self._xi_len):
    #         xi[m] = 1.0 - EFFECTIVELY_ZERO
    #         objective = self._objective_value(xi), m
    #         if objective[0] < best[0]:
    #             best = objective
    #         xi[m] = 0.0
    #     if best[1] < self._xi_len:
    #         xi[best[1]] = 1.0 - EFFECTIVELY_ZERO
    #     return xi
    #
    # # noinspection PyAttributeOutsideInit
    # @property
    # def D_jacobian(self) -> NP.Tensor3:
    #     """ Calculate the Jacobian d (``D[:, ;, m]``) / d (``xi``).
    #
    #     Returns:  The Tensor3(L, L, M-m) jacobian d(D[:, ;, m]) / d (xi).
    #     """
    #
    #     """ Calculate various jacobians."""
    #     Theta_jac = zeros((self._m + 1, self._M, self._xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
    #     Theta_jac[self._m, :, :] = transpose(self.Theta_old[self._m:self._M, :])
    #     Sigma_jac = einsum('mMj, Mk -> mkj', Theta_jac, self._Sigma_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     inv_Sigma_jac_Sigma = zeros((self._m + 1, self._m + 1, self._xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
    #     _2I_minus_Sigma_jac = einsum('mMj, Mk -> mkj', Theta_jac, self._2I_minus_Sigma_partial, optimize=True,
    #                                       dtype=float, order=self.MEMORY_LAYOUT)
    #     inv_2I_minus_Sigma_jac_2I_minus_Sigma = zeros((self._m + 1, self._m + 1, self._xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
    #     Phi_jac = einsum('mMj, Mk -> mkj', Theta_jac, self._Phi_partial, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     inv_Phi_jac_Phi = zeros((self._m + 1, self._m + 1, self._xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
    #     inv_Phi_inv_Sigma_jac_Sigma = zeros((self._m + 1, self._m + 1, self._xi_len + 1), dtype=float, order=self.MEMORY_LAYOUT)
    #     log_D_const_jac = zeros(self._xi_len + 1, dtype=float, order=self.MEMORY_LAYOUT)
    #     for j in range(self._xi_len + 1):
    #         inv_Sigma_jac_Sigma[:, :, j] = cho_solve(self._Sigma_cho, Sigma_jac[:, :, j], overwrite_b=False, check_finite=False)
    #         inv_2I_minus_Sigma_jac_2I_minus_Sigma[:, :, j] = cho_solve(self._2I_minus_Sigma_cho, _2I_minus_Sigma_jac[:, :, j],
    #                                                                          overwrite_b=False, check_finite=False)
    #         inv_Phi_jac_Phi[:, :, j] = cho_solve(self._Phi_cho, Phi_jac[:, :, j], overwrite_b=False, check_finite=False)
    #         inv_Phi_inv_Sigma_jac_Sigma[:, :, j] = cho_solve(self._Phi_cho, inv_Sigma_jac_Sigma[:, :, j], overwrite_b=False, check_finite=False)
    #         log_D_const_jac[j] = (sum(diag(inv_2I_minus_Sigma_jac_2I_minus_Sigma[:, :, j])) + sum(diag(inv_Sigma_jac_Sigma[:, :, j])))
    #
    #     """ Calculate self._V, a Tensor3(N, N, M-m,) known in the literature as V. """
    #     Sigma_factor_transpose = Theta_jac - einsum('kmj, kM -> mMj', inv_Sigma_jac_Sigma, self.Theta,
    #                                                       optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     Theta_inv_Sigma_Theta_jac = einsum('mMj, mK -> MKj', Sigma_factor_transpose, self._inv_Sigma_Theta,
    #                                        optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     T_inv_Sigma_T_jac = einsum('NM, MKj, NK -> Nj', self._FBold, Theta_inv_Sigma_Theta_jac, self._FBold,
    #                                optimize=True, dtype=float, order=self.MEMORY_LAYOUT).reshape((1, self._N, self._xi_len + 1),
    #                                                                                               order=self.MEMORY_LAYOUT)
    #     Phi_factor_transpose = Theta_jac - einsum('kmj, kM -> mMj', inv_Phi_jac_Phi, self.Theta,
    #                                               optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     Theta_inv_Phi_inv_Sigma_Theta_jac = einsum('mMj, mK -> MKj', Phi_factor_transpose, self._inv_Phi_inv_Sigma_Theta,
    #                                                optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     Theta_inv_Phi_inv_Sigma_Theta_jac -= einsum('kM, kmj, mK -> MKj', self.Theta, inv_Phi_inv_Sigma_jac_Sigma, self._inv_Sigma_Theta,
    #                                                 optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     T_inv_Phi_inv_Sigma_T_jac = einsum('NOM, MKj, NOK -> NOj',
    #                                        self._V_pre_outer_square, Theta_inv_Phi_inv_Sigma_Theta_jac, self._V_pre_outer_square,
    #                                        optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     self._V = T_inv_Phi_inv_Sigma_T_jac - (T_inv_Sigma_T_jac + transpose(T_inv_Sigma_T_jac, (1, 0, 2)))
    #
    #     """ Calculate D_jacobian. """
    #     D_derivative = self._D_const * einsum('LN, NO, NOj, KO -> LKj', self._fBold_bar_0, self._W, self._V, self._fBold_bar_0,
    #                                           optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     D_derivative -= einsum('j, LK -> LKj', log_D_const_jac, self._D_plus_Ft_1_Ft, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     result = -self._xi[1:] / self._xi[0]
    #     result = einsum('LK, j -> LKj', D_derivative[:, :, 0], result, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
    #     return result + D_derivative[:, :, 1:]

    def calculate(self):
        """ Calculate the Sobol' Indices S1 and S."""

        def _diagonal_matrices():
            """ Cache the diagonal parts of the key (Mx,Mx) matrices V, Sigma, Phi.
                _FBold_diagonal = (_lengthscale^(2) + I)^(-1)
                _Sigma_diagonal = (lengthscales^(-2) + I)^(-1)
                _Psi_diagonal = (2*lengthscales^(-2) + I) (lengthscales^(-2) + I)^(-1)
            All invariant.
            """
            self._FBold_diagonal = self._lengthscale ** 2
            self._Psi_diagonal = self._FBold_diagonal ** (-1)
            self._FBold_diagonal += 1.0
            self._Psi_diagonal += 1.0
            self._2Sigma_diagonal = 2 * (self._Psi_diagonal ** (-1))
            self._FBold_diagonal = self._FBold_diagonal ** (-1)
            self._2Psi_diagonal = (2 * self._Psi_diagonal - 1.0) * self._2Sigma_diagonal

        _diagonal_matrices()

        def _precursors_to_D():
            """ Cache invariant precursors to _D."""
            self._FBold = einsum('NM, M -> MN', self._gp.X, self._FBold_diagonal, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            self._FBold_outer_plus = self._FBold_outer_minus = self._FBold.reshape((self._M, 1, self._N))
            FBold_T = transpose(self._FBold_outer_minus, (0, 2, 1))
            self._FBold_outer_plus = self._FBold_outer_plus + FBold_T
            self._FBold_outer_minus = self._FBold_outer_minus - FBold_T

        _precursors_to_D()

        def _conditional_expectations():
            """ Cache invariant conditional expectations _fBold_bar_0 and _f_bar_0_2. """
            self._fBold_bar_0 = einsum('MN, NM -> N', self._FBold, self._gp.X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            self._fBold_bar_0 = sqrt(prod(self._2Sigma_diagonal) / (2 ** self._M)) * exp(-0.5 * self._fBold_bar_0)
            self._fBold_bar_0 = einsum('Ll, Nl, N -> LN', self._gp.parameters.f, self._gp.inv_prior_Y_Y, self._fBold_bar_0, optimize=True,
                                       dtype=float, order=self.MEMORY_LAYOUT)
            self._f_bar_0_2 = einsum('LN -> L', self._fBold_bar_0, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            self._f_bar_0_2 = einsum('L, l -> Ll', self._f_bar_0_2, self._f_bar_0_2, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)

        _conditional_expectations()

        def _DBold_DBold_full_DBold_full_determinant() -> Tuple[NP.Matrix, NP.Tensor3, NP.Array]:
            """ Calculate DBold and its expanded form DBold_full, with a determinant pre-factor. """
            DBold_full_determinant = (self._2Sigma_diagonal / self._2Psi_diagonal) ** (1 / 2)
            DBold_numerator_precision_halved = 0.5 * (self._2Sigma_diagonal ** (-1))
            DBold_denominator_precision_halved = 0.5 * (self._2Psi_diagonal ** (-1))
            DBold_full = einsum('MNO, M, MNO -> MNO', self._FBold_outer_minus, DBold_numerator_precision_halved, self._FBold_outer_minus,
                                optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            DBold_full -= einsum('MNO, M, MNO -> MNO', self._FBold_outer_plus, DBold_denominator_precision_halved, self._FBold_outer_plus,
                                 optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            DBold = einsum('MNO -> NO', DBold_full, optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
            return ((prod(DBold_full_determinant) ** (-1)) * exp(-DBold), exp(DBold_full), DBold_full_determinant)

        DBold, DBold_full, DBold_full_determinant = _DBold_DBold_full_DBold_full_determinant()

        def sobol_and_reorder() -> NP.Array:
            """ Calculate all Sobol Indices, returning a reordering by decreasing ST_m """
            self._D[:, :, -1] = einsum('LN, NO, lO -> Ll', self._fBold_bar_0, DBold, self._fBold_bar_0,
                                       optimize=True, dtype=float, order=self.MEMORY_LAYOUT) - self._f_bar_0_2
            DBold_m_included = DBold.copy(order=self.MEMORY_LAYOUT)
            DBold_m_to_include = empty((self._N, self._N), dtype=float, order=self.MEMORY_LAYOUT)
            DBold_m = empty((self._N, self._N), dtype=float, order=self.MEMORY_LAYOUT)
            reordering = -ones(self._M, dtype=int, order=self.MEMORY_LAYOUT)
            D_m = empty((self._L, self._L), dtype=float, order=self.MEMORY_LAYOUT)
            S_m = empty((self._L, self._L), dtype=float, order=self.MEMORY_LAYOUT)
            for m_excluded in reversed(range(1, self._M)):
                max_semi_norm = (-1, -1.5)
                for m_to_exclude in reversed(range(self._M)):
                    if m_to_exclude not in reordering[m_excluded:]:
                        DBold_m[:] = DBold_m_included * DBold_full[m_to_exclude, :, :] * DBold_full_determinant[m_to_exclude]
                        einsum('LN, NO, lO -> Ll', self._fBold_bar_0, DBold_m, self._fBold_bar_0, out=D_m,
                               optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
                        D_m -= self._f_bar_0_2
                        S_m[:] = D_m / self._D[:, :, -1]
                        semi_norm = self._semi_norm.value(S_m)
                        if semi_norm > max_semi_norm[1]:
                            max_semi_norm = (m_to_exclude, semi_norm)
                            copyto(DBold_m_to_include, DBold_m)
                            self._D[:, :, m_excluded - 1] = D_m[:]
                reordering[m_excluded] = max_semi_norm[0]
                self._S[:, :, m_excluded - 1] = self._D[:, :, m_excluded - 1] / self._D[:, :, -1]
                self._ST[:, :, m_excluded] = 1 - self._S[:, :, m_excluded - 1]
                DBold_m = (DBold_full[max_semi_norm[0], :, :] * DBold_full_determinant[max_semi_norm[0]]) ** (-1)
                einsum('LN, NO, lO -> Ll', self._fBold_bar_0, DBold_m, self._fBold_bar_0, out=D_m,
                       optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
                D_m -= self._f_bar_0_2
                self._S1[:, :, m_excluded] = D_m / self._D[:, :, -1]
                copyto(DBold_m_included, DBold_m_to_include)
            self._S1[:, :, 0] = self._S[:, :, 0]
            reordering[0] = (set(range(self._M)) - set(reordering)).pop()  # This is the element in range(self._M) which is missing from reordering.
            return reordering

        reordering = sobol_and_reorder()

        self.reorder_data_columns(reordering)
        theta = eye(self._M, dtype=float, order=self.MEMORY_LAYOUT)
        self.write_parameters(self.Parameters(theta[reordering, :], self.Tensor3AsMatrix(self._D),
                                              self.Tensor3AsMatrix(self._S1), self.Tensor3AsMatrix(self._S), self.Tensor3AsMatrix(self._ST)))

    def _read_semi_norm(self, semi_norm_meta: Dict) -> Sobol.SemiNorm:
        # noinspection PyTypeChecker
        semi_norm_json = self._folder / "SemiNorm.json"
        if semi_norm_meta is None:
            if self._options_json.exists():
                with open(semi_norm_json, mode='r') as file:
                    semi_norm_meta = json.load(file)
            else:
                semi_norm_meta = self.DEFAULT_OPTIMIZER_OPTIONS
        if semi_norm_meta['L'] != self._L:
            warn("I am changing Sobol.semi_norm.meta['L'] from {0:d} to {1:d} = Sobol.gp.L.".format(semi_norm_meta['L'], self._L))
            semi_norm_meta['L'] = self._L
        with open(semi_norm_json, mode='w') as file:
            json.dump(semi_norm_meta, file, indent=8)
        return Sobol.SemiNorm.from_meta(semi_norm_meta)

    def __init__(self, gp: GP, semi_norm: Dict = SemiNorm.DEFAULT_META):
        """ Initialize Sobol' Calculator and Optimizer.

        Args:
            gp: The underlying Gaussian process surrogate.
            semi_norm: Meta json describing a Sobol.SemiNorm.
        """

        """ Private Attributes:
        _gp (invariant): The underlying GP.
        _N, _M, _L (invariant): _gp.N, _gp.M, _gp.L.
        _lengthscale (invariant): The (M,) Array _gp.kernel_parameters.parameters.lengthscales[0, :].
        _FBold_diagonal, _2Sigma_diagonal, _2Psi_diagonal (invariant): Arrays of shape (M,) representing (M,M) diagonal matrices.

        _fBold_bar_0 (invariant): An (L,N) Matrix.
        _f_bar_0_2 (invariant): The (L,L) Matrix product of E[f] E[f.T].
        _FBold (invariant): An (M,N) Matrix.

        _objective_value: The optimization objective value (function of Theta_M_M), set by the call to Sobol.optimize()
        _objective_jacobian: The optimization objective jacobian (function of Theta_M_M), set by the call to Sobol.optimize()

        _Sigma_tilde, _Psi_tilde: (M,M) matrices.
        _FBold_tilde: An (M,N) matrix.
        """

        """ Initialize surrogate GP and related quantities, namely 
            _gp
            _N, _M, _L,: GP training data dimensions N = dataset rows (datapoints), M = input columns, L = input columns
            _lengthscale: RBF lengthscales vector
        all of which are private and invariant.
        """
        self._gp = gp
        self._N, self._M, self._L = self._gp.N, self._gp.M, self._gp.L
        self._lengthscale = self._gp.kernel.parameters.lengthscales[0, :]
        if self._lengthscale.shape != (self._M,):
            self._lengthscale = full(self._M, self._lengthscale[0], dtype=float, order=self.MEMORY_LAYOUT)

        self._Theta_M_M = zeros(self._M, dtype=float, order=self.MEMORY_LAYOUT)
        self._D = empty((self._L, self._L, self._M), dtype=float, order=self.MEMORY_LAYOUT)
        self._S1 = empty((self._L, self._L, self._M), dtype=float, order=self.MEMORY_LAYOUT)
        self._S = ones((self._L, self._L, self._M), dtype=float, order=self.MEMORY_LAYOUT)
        self._ST = zeros((self._L, self._L, self._M), dtype=float, order=self.MEMORY_LAYOUT)

        """ Declare internal calculation stages. These are documented where they are calculated, in Sobol.calculate()."""
        self._FBold_diagonal = self._2Sigma_diagonal = self._2Psi_diagonal = None
        self._FBold = self._V_pre_outer_square = self._fBold_bar_0 = self._f_bar_0_2 = None
        self._objective_value = self._objective_jacobian = None

        super().__init__(self._gp.folder / self.NAME, self.Parameters(eye(self._M, dtype=float, order=self.MEMORY_LAYOUT),
                                                                      self.Tensor3AsMatrix(self._D), self.Tensor3AsMatrix(self._S1),
                                                                      self.Tensor3AsMatrix(self._S), self.Tensor3AsMatrix(self._ST)))
        self._semi_norm = self._read_semi_norm(semi_norm)
        self.calculate()

    @classmethod
    @abstractmethod
    def from_GP(cls, fold: Fold, source_gp_name: str, destination_gp_name: str, semi_norm: Dict = SemiNorm.DEFAULT_META) -> Sobol:
        """ Create a Sobol object from a saved GP folder.

        Args:
            fold: The Fold housing the source and destination GPs.
            source_gp_name: The source GP folder.
            destination_gp_name: The destination GP folder. Must not exist.
            semi_norm: Meta json describing a Sobol.SemiNorm.
        Returns: The constructed Sobol object
        """
        dst = fold.folder / destination_gp_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src=fold.folder / source_gp_name, dst=dst)
        return cls(gp=GP(fold=fold, name=destination_gp_name, semi_norm=semi_norm))
