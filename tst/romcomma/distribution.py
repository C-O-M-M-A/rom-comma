# BSD 3-Clause License
#
# Copyright (c) 2019, Robert A. Milton
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Contains basic probability distributions, with a view to sampling.

**Contents**:
    **SampleDesign** enum expressing the sample design to use.

    **Univariate** class encapsulating a univariate, fully parametrized (SciPy frozen) continuous distribution.

    **Multivariate** container class.

    **Multivariate.Base** abstract interface.

    **Multivariate.Independent(Multivariate.Base)** class encapsulating a multivariate distribution composed of independent marginals.

    **Multivariate.Normal(Multivariate.Base)** class encapsulating a multivariate normal distribution.
"""

from romcomma.typing_ import Any, Optional, Union, NP, Sequence, Tuple
from scipy import linalg, stats
from pyDOE import lhs as pydoe_lhs
from numpy import zeros, atleast_2d
from enum import Enum, auto
from abc import ABC, abstractmethod


class SampleDesign(Enum):
    RANDOM_VARIATE = auto()
    LATIN_HYPERCUBE = auto()


# noinspection PyPep8Naming
class Univariate:
    """ A univariate, fully parametrized (SciPy frozen) continuous distribution."""

    SuperFamily = stats.rv_continuous
    """ The SciPy (super) family of continuous, univariate distributions."""

    Family = {_class.__name__[:-4]: _class for
              _class in SuperFamily.__subclasses__() if _class.__name__[-4:] == "_gen"}
    """ The romcomma dictionary of univariate distribution classes."""
    family = {name: Fam() for name, Fam in Family.items()}
    """ The romcomma dictionary of unparametrized univariate distribution objects."""
    names = Family.keys()
    """ The romcomma list of unparametrized univariate distributions."""

    @classmethod
    def parameter_defaults(cls, name: str) -> dict:
        """ List the default values for the parameters of the named distribution.

        Args:
            name: The name of the univariate distribution to interrogate.
        Returns: A Dict of parameters and their default values.
        """
        keys = cls.family[name].shapes.split(',') if cls.family[name].shapes else []
        return {**{'name': name, "loc": 0, "scale": 1}, **{_key: None for _key in keys}}

    @property
    def name(self) -> str:
        """ The name of this univariate distribution."""
        return self._name

    @property
    def parameters(self) -> dict:
        """ The parameters of this univariate distribution."""
        return self._parameters

    @property
    def parametrized(self) -> SuperFamily:
        """ The SciPy parametrized univariate distribution."""
        return self._parametrized

    @property
    def unparametrized(self) -> SuperFamily:
        """ A **new** instance of the unparametrized distribution.
        An existing unparametrized instance is always available through Univariate.family[name].
        """
        return Univariate.Family[self._name]()

    def rvs(self, N: int, M: int) -> NP.Vector:
        """ Random variate sample from this distribution.Univariate.

        Args:
            N: The number of rows returned.
            M: The number of columns returned.
        Returns: An NxM Matrix of random noise sampled from the underlying distribution.Univariate.

        Raises:
            ValueError: If N &lt 1.
            ValueError: If M &lt 1.
        """
        if N < 1:
            raise ValueError("N == {N:d} < 1".format(N=N))
        if M < 1:
            raise ValueError("M == {M:d} < 1".format(M=M))
        return self.parametrized.rvs(size=[N, M])

    def __init__(self, name: str = "uniform", **kwargs: Any):
        """ Construct a Univariate distibution.

        Args:
            name: The name of the underlying Unparametrized distribution, e.g. "uniform" or "norm".
                Univariate.names() lists the admissible names.
            **kwargs: The parameters passed directly to the Univariate.Family[name] (class) constructor.
                Univariate.parameters(name) supplies the kwargs Dict required.
        """
        self._name = name
        self._parametrized = self.unparametrized.freeze(**kwargs)
        self._parameters = {**Univariate.parameter_defaults(self._name), **kwargs}


class Multivariate:
    """ Container class for multivariate distributions."""
    # noinspection PyPep8Naming,PyPep8Naming
    class Base(ABC):
        """ Absatract interface. """

        @property
        def M(self) -> int:
            """ The dimesionality of the multivariate."""
            return self._M

        @property
        @abstractmethod
        def parameters(self) -> dict:
            """ The parameters of this multivariate distribution, as a Dict containing ``M`` and ``marginals`` which is a list of parameter 
            Dicts for each marginal distribution in turn."""

        @abstractmethod
        def sample(self, N: int, sample_design: SampleDesign = SampleDesign.LATIN_HYPERCUBE) -> NP.Matrix:
            """ Sample random noise from this multivariate.

            Args:
                N: The number of rows returned.
                sample_design: A SampleDesign, either LATIN_HYPERCUBE or RANDOM_VARIATE. Defaults to LATIN_HYPERCUBE.
            Returns: A Matrix(N, self.M) of random noise sampled from the underlying Multivariate.Base.
            """

        @abstractmethod
        def __init__(self):
            """ This is actually inaccessible, but serves as an abstract declaration."""
            self._M = 0

    # noinspection PyPep8Naming
    class Independent(Base):
        """ A multivariate distribution composed of independent marginals."""

        @property
        def marginals(self) -> Tuple[Univariate]:
            """ A Tuple of univariate marginals. The Tuple is empty if all marginals are identical to the standard uniform distribution ~U(0,1).
            The Tuple is singleton if all marginals are identical (iid). Otherwise the Tuple length is equal to M, comprising one univariate 
            marginal per X-dimension.
            """
            return self._marginals if self._marginals else tuple([Univariate(name="uniform", loc=0, scale=1)])

        @property
        def parameters(self) -> dict:
            """ The parameters of this multivariate distribution, as a Dict containing ``M`` and ``marginals`` which is a list of parameter 
            Dicts for each marginal distribution in turn."""
            marginals = [marginal.parameters for marginal in self.marginals]
            return {"M": self._M, "marginals": marginals}

        def lhs(self, N: int, criterion: Optional[str] = None, iterations: Optional[int] = None) -> NP.Matrix:
            """ Sample latin hypercube noise from this Multivariate.Independent.

            Args:
                N: The number of sample points to generate.
                criterion: Allowable values are "center" or "c", "maximin" or "M", "centermaximin"
                    or "cm", and "correlation" or "corr". If no value is given, the design is simply
                    randomized. For further details see https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube.
                iterations: The number of iterations in the maximin and correlations algorithms (Default: 5).
            Returns: An (N, M) latin hypercube design Matrix.

            Raises:
                ValueError: If N &lt 1.
            """
            if N < 1:
                raise ValueError("N = {N:d} < 1".format(N=N))
            _lhs = pydoe_lhs(self._M, N, criterion, iterations)
            if len(self._marginals) > 1:
                for i in range(self._M):
                    _lhs[:, i] = self._marginals[i].parametrized.ppf(_lhs[:, i])
            elif len(self._marginals) == 1:
                for i in range(self._M):
                    _lhs[:, i] = self._marginals[0].parametrized.ppf(_lhs[:, i])
            return _lhs

        def rvs(self, N: int) -> NP.Matrix:
            """ Sample random noise from this Multivariate.Independent.
            
            Args:
                N: The number of rows returned.
            Returns: An (N,M) design Matrix of random noise sampled from the underlying Multivariate.Independent.

            Raises:
                ValueError: If N &lt 1.
            """
            if N < 1:
                raise ValueError("N == {N:d} < 1".format(N=N))
            if len(self._marginals) > 1:
                result = zeros([N, self.M])
                for i in range(self._M):
                    result[:, [i]] += self._marginals[i].rvs(N, 1)
                return result
            else:
                if len(self._marginals) == 1:
                    return self._marginals[0].rvs(N, self.M)
                else:
                    return Univariate(name="uniform", loc=0, scale=1).rvs(N, self.M)

        def cdf(self, X: NP.Matrix) -> NP.Matrix:
            """ Calculate cumulative distribution function (CDF) of this Multivariate.Independent.

            Args:
                X: NxM Matrix of values.
            Returns:  MxN Matrix of CDF(values)

            Raises:
                ValueError: If len(X.shape) != 2 or X.shape[1] != self.M.
            """
            if len(X.shape) != 2 or X.shape[1] != self.M:
                raise ValueError("X.shape = {s} when M ={M:d}".format(s=X.shape, M=self.M))
            N = X.shape[0]
            if len(self._marginals) > 1:
                result = zeros([N, self.M])
                for i in range(self._M):
                    result[:, [i]] += self._marginals[i].parametrized.cdf(X[:, [i]])
                return result
            else:
                if len(self._marginals) == 1:
                    return self._marginals[0].parametrized.cdf(X)
                else:
                    return Univariate(name="uniform", loc=0, scale=1).parametrized.cdf(X)

        def sample(self, N: int, sample_design: SampleDesign = SampleDesign.LATIN_HYPERCUBE) -> NP.Matrix:
            """ Sample random noise from this Multivariate.Independent.

            Args:
                N: The number of rows returned.
                sample_design: A SampleDesign, either LATIN_HYPERCUBE or RANDOM_VARIATE. Defaults to LATIN_HYPERCUBE.
            Returns: An (N,M) design Matrix of random noise sampled from the underlying Multivariate.Independent.
            """
            return self.rvs(N) if sample_design is SampleDesign.RANDOM_VARIATE else self.lhs(N)

        # noinspection PyMissingConstructor
        def __init__(self, M: int = 0, marginals: Union[Univariate, Sequence[Univariate]] = tuple()):
            """ Construct a Multivariate.Independent distribution.

            Args:
                M: The dimensionality (number of factors) of the multivariate.
                    If non-positive the value used is inferred from the supplied marginals.
                marginals: A Tuple of distribution.univariate marginals defining the multivariate.
                    If this Tuple is empty all marginals are identical (iid) to the standard uniform
                    distribution ~U(0,1). If this Tuple is singleton all marginals are identical (iid) to the one given.
                    Otherwise this Tuple must be of length M, comprising one univariate marginal per dimension (factor).

            Raises:
                TypeError: If neither M nor marginals are provided.
                ValueError: If M and marginals are both provided, but marginal is not of length M.
            """
            if not marginals:
                if M < 1:
                    raise TypeError("Neither M nor marginals were provided.")
                else:
                    self._M = M
                    self._marginals = tuple()
            else:
                self._marginals = (marginals,) if isinstance(marginals, Univariate) else tuple(marginals)
                if 0 >= M:
                    self._M = len(self._marginals)
                else:
                    self._M = M
                    if 1 < len(self.marginals) != self._M:
                        raise ValueError("M does not match len(marginals).")

    # noinspection PyPep8Naming
    class Normal(Base):
        """ A multivariate normal distribution."""

        @property
        def mean(self) -> NP.Covector:
            """ Distribution mean, as a (1,M) Covector."""
            return self._mean

        @property
        def covariance(self) -> NP.Matrix:
            """ Distribution covariance, as a (M,M) Matrix."""
            return self._covariance

        @property
        def cholesky(self) -> NP.Matrix:
            """ The upper triangular Cholesky factor of ``self.covariance``."""
            if self._cholesky is None:
                self._cholesky = linalg.cholesky(self._covariance, lower=False, overwrite_a=False, check_finite=False)
            return self._cholesky

        @property
        def eigen(self) -> Tuple[NP.Vector, NP.Matrix]:
            """ The eigensystem of ``self.covariance``, in the form (Vector of eigenvalues, Matrix of eigenvectors)."""
            if self._eigen is None:
                self._eigen = linalg.eigh(self._covariance, b=None, lower=False, eigvals_only=False, overwrite_a=False, overwrite_b=False,
                                          turbo=True, eigvals=None, type=1, check_finite=False)
            return self._eigen

        @property
        def parameters(self) -> dict:
            """ The parameters of this multivariate distribution, as a Dict containing ``M`` and ``marginals`` which is a list of parameter 
            Dicts for each marginal distribution in turn."""
            return {"M": self._M, "name": "Multivariate.Normal", "mean": self._mean.tolist(), "covariance": self._covariance.tolist()}

        def sample(self, N: int, sample_design: SampleDesign = SampleDesign.LATIN_HYPERCUBE) -> NP.Matrix:
            """ Sample random noise from this Multivariate.Independent.

            Args:
                N: The number of rows returned.
                sample_design: A SampleDesign, either LATIN_HYPERCUBE or RANDOM_VARIATE. Defaults to LATIN_HYPERCUBE.
            Returns: An (N,M) design Matrix of random noise sampled from the underlying Multivariate.Normal.
            """
            iid_standard_normal_sample = self._iid_standard_normal.sample(N, sample_design)
            return self.mean + iid_standard_normal_sample @ self.cholesky

        # noinspection PyMissingConstructor
        def __init__(self, mean: NP.Covector, covariance: NP.MatrixLike):
            """ Construct a Multivariate.Normal distribution

            Args:
                mean: A (1,M) CoVector of means
                covariance: An (M,M) covariance Matrix
            """
            self._mean = atleast_2d(mean)
            if not (1 == self._mean.shape[0] <= self._mean.shape[1]):
                raise TypeError("Mean is not CovectorLike.")
            self._M = self._mean.shape[1]
            self._covariance = atleast_2d(covariance)
            if not (self._M == self._covariance.shape[0] == self._covariance.shape[1]):
                raise TypeError("Covariance is not MatrixLike.")
            self._cholesky = None
            self._eigen = None
            self._iid_standard_normal = Multivariate.Independent(self._M, Univariate("norm", loc=0, scale=1))
