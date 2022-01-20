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
from romcomma.base.classes import Parameters
from romcomma.gpr.models import GPInterface


LOG2PI = tf.math.log(tf.constant(2 * np.pi, dtype=FLOAT()))
Triad = Dict[str, TF.Tensor]
TriadOfTriads = Dict[str, Triad]


def gaussian(mean: TF.Tensor, variance: TF.Tensor, is_variance_diagonal: bool,
             ordinate: TF.Tensor = tf.constant(0, dtype=FLOAT())) -> TF.Tensor:
    """ Computes the gaussian probability density.

    Args:
        mean: Gaussian population mean.
        variance: The lower triangular Cholesky decomposition of the Gaussian population variance.
        is_variance_diagonal: True if variance is an m-vector
        ordinate: The ordinate (z-value) to calculate the Gaussian density for.
    Returns: The Gaussian pdf.
    """
    ordinate = ordinate - mean if ordinate.shape else mean
    result = ordinate / variance if is_variance_diagonal else tf.linalg.triangular_solve(variance, ordinate, lower=True)
    result = tf.einsum('...o, ...o -> ...', result, result)
    result = result + tf.reduce_prod(variance) if is_variance_diagonal else tf.reduce_prod(tf.linalg.diag_part(variance))
    return tf.math.exp(-0.5 * (LOG2PI * ordinate.shape[-1] + result))


class GSAInterface(gf.Module):
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
        """ Calculation options. ``is_T_partial`` forces W[`m`][`M`] = W[`M`][`M`] = 0."""
        return {'is_S_diagonal': False, 'is_T_calculated': True, 'is_T_diagonal': False, 'is_T_partial': False}

    @classmethod
    @property
    def NaN(cls) -> TF.Tensor:
        return tf.constant(np.NaN, dtype=FLOAT())

    @classmethod
    def Triad(cls) -> Dict[str, Union[Triad, TF.Tensor]]:
        """ Data structure for a Triad, populated with NaNs."""
        return {'0': GSA.NaN, 'm': GSA.NaN, 'M': GSA.NaN}

    @property
    def S(self) -> TF.Tensor:
        """ The closed Sobol index tensor shaped (L,) if ``is_S_diagonal`` else (L, L)."""
        return self._S['m'] if self._m > 0 else GSA.NaN

    @property
    def T(self) -> TF.Tensor:
        """ The cross covariance of S, shaped (L,) (if ``is_S_diagonal and is_T_diagonal``),
        (L,L)  (if ``is_S_diagonal xor is_T_diagonal``), or (L,L,L,L)  (if ``is_S_diagonal nor is_T_diagonal``). """
        return self._T['m'] if self._m > 0 else GSA.NaN

    @property
    def V(self) -> Triad:
        """ The conditional variance Triad. Each element is shaped (L,) if ``is_S_diagonal`` else (L, L)."""
        return self._V if self._m > 0 else GSA.Triad()

    def W(self) -> TriadOfTriads:
        """ The cross covariance of V. Each element is shaped (L,) (if ``is_S_diagonal and is_T_diagonal``),
        (L,L)  (if ``is_S_diagonal xor is_T_diagonal``), or (L,L,L,L)  (if ``is_S_diagonal nor is_T_diagonal``). """
        return self._W if self._m > 0 else GSA.Triad()

    @property
    def Theta(self) -> TF.Matrix:
        """ The input rotation matrix."""
        return tf.eye(self.M, dtype=FLOAT()) if self._Theta == GSA.NaN else self._Theta

    @Theta.setter
    def Theta(self, value: TF.Matrix):
        self._Theta = value
        self._calculate_without_marginalizing(**self.options)
        self._m = 0

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
        tf.debugging.assert_type(value, INT())
        tf.assert_greater(value, 0, f'm={value} must be between 0 and M+1={self.M+1}.')
        tf.assert_less(value, self.M + 1, f'm={value} must be between 0 and M+1={self.M+1}.')
        if self._m != value:
            self._m = value
            self._marginalize(**self.options)

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
    def _marginalize_M0(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        """ Calculate the closed Sobol index tensors S['M'], S.['0'], and, optionally, their cross covariances T['M'], T['0'].

        Args:
            is_S_diagonal: If True, only the L-diagonal (variance) elements of S and V are calculated.
            is_T_calculated: If False, T and W return NaN.
            is_T_diagonal: If True, only the S.shape-diagonal elements of T and W are calculated.
                In other words the variance of each element of S is calculated, but cross-covariances are not.
            is_T_partial: If True this effectively forces W[`m`][`M`] = W[`M`][`M`] = 0, as if the full ['M'] model is variance free.
        """
        raise NotImplementedError

    @abstractmethod
    def _marginalize(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        """ Calculate the closed Sobol index tensor S['m'] of the first m input dimensions, and, optionally, its cross covariance T['m'].

        Args:
            is_S_diagonal: If True, only the L-diagonal (variance) elements of S and V are calculated.
            is_T_calculated: If False, T and W return NaN.
            is_T_diagonal: If True, only the S.shape-diagonal elements of T and W are calculated.
                In other words the variance of each element of S is calculated, but cross-covariances are not.
            is_T_partial: If True this effectively forces W[`m`][`M`] = W[`M`][`M`] = 0, as if the full ['M'] model is variance free.
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_without_marginalizing_M0(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        """ Calculate all quantities up to marginalization. Essentially, this calculates all the means and variances of all Gaussians behind the GSA,
        but stops short of calculating the (marginalized) Gaussians.

        Args:
            is_S_diagonal: If True, only the L-diagonal (variance) elements of S and V are calculated.
            is_T_calculated: If False, T and W return NaN.
            is_T_diagonal: If True, only the S.shape-diagonal elements of T and W are calculated.
                In other words the variance of each element of S is calculated, but cross-covariances are not.
            is_T_partial: If True this effectively forces W[`m`][`M`] = W[`M`][`M`] = 0, as if the full ['M'] model is variance free.
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_without_marginalizing(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        """ Calculate all quantities up to marginalization. Essentially, this calculates all the means and variances of all Gaussians behind the GSA,
        but stops short of calculating the (marginalized) Gaussians.

        Args:
            is_S_diagonal: If True, only the L-diagonal (variance) elements of S and V are calculated.
            is_T_calculated: If False, T and W return NaN.
            is_T_diagonal: If True, only the S.shape-diagonal elements of T and W are calculated.
                In other words the variance of each element of S is calculated, but cross-covariances are not.
            is_T_partial: If True this effectively forces W[`m`][`M`] = W[`M`][`M`] = 0, as if the full ['M'] model is variance free.
        """
        raise NotImplementedError

    def _calculate_M0(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        """ Calculate all quantities for m=M and m=0, setting Theta=None and m=0 for this GSA.

        Args:
            is_S_diagonal: If True, only the L-diagonal (variance) elements of S and V are calculated.
            is_T_calculated: If False, T and W return NaN.
            is_T_diagonal: If True, only the S.shape-diagonal elements of T and W are calculated.
                In other words the variance of each element of S is calculated, but cross-covariances are not.
            is_T_partial: If True this effectively forces W[`m`][`M`] = W[`M`][`M`] = 0, as if the full ['M'] model is variance free.
        """
        self._Theta = None
        self._calculate_without_marginalizing_M0(**self.options)
        self._marginalize_M0(**self.options)
        self._m = tf.constant(0, dtype=INT())

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
        super(GSAInterface, self).__init__(name=name)
        self._gp = gp
        self._folder = self._gp.folder / 'gsa' / name
        self.options = self.OPTIONS | kwargs
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
        self._Theta = GSA.NaN
        self._m = tf.constant(0, dtype=INT())
        self._calculate_M0(**self.options)

        
class GSA(GSAInterface):
    """ Implementation of Global Sensitivity Analysis."""

    def _marginalize_M0(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        self._m = self.M
        self._V_M0(is_S_diagonal, is_T_calculated, is_T_diagonal, is_T_partial)
        self._S = GSA.Triad()
        self._S['0'] = tf.zeros_like(self._V['0'])
        self._S['M'] = tf.ones_like(self._V['M'])
        self._T = GSA.Triad()
        self._W_M0(is_S_diagonal, is_T_calculated, is_T_diagonal, is_T_partial)
        if is_T_calculated:
            self._T['0'] = tf.zeros_like(self._W['M'])
            self._T['M'] = tf.zeros_like(self._W['M'])

    def _V_M0(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        self._V = GSA.Triad()
        if is_S_diagonal:
            self._V['0'] = tf.zeros(shape=(self.L,), dtype=FLOAT())
            self._V['M'] = 0.5 * tf.ones(shape=(self.L,), dtype=FLOAT())
            self._V_ein = tf.constant(['L','J'])
        else:
            self._V['0'] = tf.zeros(shape=(self.L, self.L), dtype=FLOAT())
            self._V['M'] = 0.5 * tf.ones(shape=(self.L, self.L), dtype=FLOAT()) - self._V['0']
            self._V_ein = tf.constant(['Ll', 'Jj'])

    def _W_M0(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        self._W = GSA.Triad()
        if is_T_calculated:
            if is_T_diagonal:
                shape = self._V['M'].shape
                self._VV_ein = tf.strings.join([self._V_ein[0], ',', self._V_ein[0], '->', self._V_ein[0]])
            else:
                shape = tf.concat([self._V['M'].shape, self._V['M'].shape], 0)
                self._VV_ein = tf.strings.join([self._V_ein[0], ',', self._V_ein[1], '->', self._V_ein[0], self._V_ein[1]])
            self._W['M'] = 0.5 * tf.ones(shape=shape, dtype=FLOAT())
            self._VV = GSA.Triad()
            self._VV['m'] = GSA.Triad()
            self._VV['M'] = tf.einsum(self._VV_ein.numpy().decode(), self._V['M'], self._V['M'])


    def _marginalize(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        self._V_m(is_S_diagonal, is_T_calculated, is_T_diagonal, is_T_partial)
        self._S['m'] = self._V['m'] / self._V['M']
        if is_T_calculated:
            self._W_m(is_S_diagonal, is_T_calculated, is_T_diagonal, is_T_partial)
            self._T['m'] = self._W['m']['m'] / self._VV['M']
            if not is_T_partial:
                self._VV['m']['m'] = tf.einsum(self._VV_ein.numpy().decode(), self._V['m'], self._V['m'])
                self._VV['m']['M'] = tf.einsum(self._VV_ein.numpy().decode(), self._V['m'], self._V['M'])
                self._T['m'] = self._T['m'] + (self._VV['m']['m'] / self._VV['M']) * (self._W['M'] / self._VV['M']
                                                                                      - 2 * self._W['m']['M'] / (self._VV['m']['M']))

    def _V_m(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        self._V['m'] = 0.5 * tf.ones_like(self._V['M'])

    def _W_m(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        self._W['m'] = GSA.Triad()
        self._W['m']['m'] = tf.ones_like(self._W['M'])
        self._W['m']['M'] = tf.ones_like(self._W['M'])

    def _calculate_without_marginalizing_M0(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        pass

    def _calculate_without_marginalizing(self, is_S_diagonal: bool, is_T_calculated: bool, is_T_diagonal: bool, is_T_partial: bool):
        raise NotImplementedError
