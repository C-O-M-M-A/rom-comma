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

""" A suite of functions designed to test_data screening (order reduction) with high-dimensional distributions.

All functions herein are precisely as described in https://www.sfu.ca/~ssurjano/screen.html.
Each function signature follows the format:

|
``def function_(X: MatrixLike, **kwargs)``

    :X: The function argument, in the form of an ``(N,M)`` design Matrix.
    :**kwargs: Function-specific parameters, normally fixed.

Returns: A ``Vector[0 : N-1, 1]`` evaluating ``function_(X[0 : N-1, :])``.

|
"""

from __future__ import annotations

from romcomma.typing_ import *
from numpy import atleast_2d, arange, prod, sin, einsum, concatenate, ndarray, array, pi
from pandas import DataFrame, MultiIndex
from romcomma.data import Store
from romcomma.test import sampling
from SALib.test_functions import Ishigami, Sobol_G, linear_model_1, linear_model_2, lake_problem


def sample(functions: Sequence[FunctionWithParameters[NP.Vector]], N: int, M: int, noise_variance:NP.MatrixLike, folder: PathLike,
           sampling_method: Callable[[int, int, Any], NP.Matrix] = sampling.latin_hypercube, **kwargs) -> Store:
    """ Store a sample of test function responses.

    Args:
        functions: A sequence of test functions, of length L.
        N: The number of samples (datapoints), N &gt 0.
        M: The input dimensionality, M &ge 0.
        noise_variance: A noise (co)variance of shape (L,L) or (L,). The latter is interpreted as an (L,L) diagonal matrix.
            Used to generate N random samples of Gaussian noise ~ N(0, noise_variance).
        folder: The Store.folder to create and store the results in.
        sampling_method: A Callable sampling_method(N, M, **kwargs) -> X, which returns an (N,M) matrix.
        kwargs: Passed directly to sampling_method.
    Returns: A store containing N rows of M input columns and L output columns. The output is f(X) + noise.
    """
    X = sampling_method(N, M, **kwargs)

    noise = sampling.multivariate_gaussian_noise(N, noise_variance)
    return apply(functions, X, noise, folder)


def apply(functions: Sequence[FunctionWithParameters[NP.Vector]], X: NP.Matrix, noise: Optional[NP.Matrix], folder: PathLike, **kwargs) -> Store:
    """ Store a sample of test function responses.

    Args:
        functions: A sequence of test functions, of length L.
        X: An (N,M) design matrix of inputs.
        noise: An (N,L) design matrix of output noise.
        folder: The Store.folder to create and store the results in.
    Returns: A store containing N rows of M input columns and L output columns. The output is f(X) + noise.
    Raises: IndexError if dimensions are incompatible.
    """
    X, noise = atleast_2d(X), atleast_2d(noise)
    if X.shape[1] == 0:
        pass    # Noise only!
    else:


    X = X_distribution.sample(N, X_sample_design)
    meta = {'origin': {'N': N, 'X_distribution': X_distribution.parameters, 'X_sample_design': str(X_sample_design),
                       'CDF_scale': atleast_2d(CDF_scale).tolist(), 'CDF_loc': atleast_2d(CDF_loc).tolist(),
                       'input_transform': None if input_transform is None else input_transform.meta,
                       'functions': None if functions is None else [f.meta for f in functions],
                       'noise_distribution': None, 'noise_sample_design': str(noise_sample_design)}}

    def __CDF_representation() -> NP.Matrix:
        return X.copy() if (CDF_loc is None or CDF_scale is None) else CDF_scale * X_distribution.cdf(X) - CDF_loc

    def __input_transform_representation(X_: NP.Matrix) -> NP.Matrix:
        return X_ if input_transform is None else input_transform(X_)

    def __functions_representation(X_: NP.Matrix) -> NP.Matrix:
        return X_ if functions is None else concatenate([f(X_) for f in functions], axis=1)

    Y = __functions_representation(__input_transform_representation(__CDF_representation()))
    if noise_distribution is not None:
        if noise_distribution.M != Y.shape[1]:
            raise IndexError(f'noise.M = {noise_distribution.M:d} != {Y.shape[1]:d} = L = dimension output.')
        Y += noise_distribution.sample(N, noise_sample_design)
        meta['origin']['noise_distribution'] = noise_distribution.parameters
    columns = [('X', f'X[{i:d}]') for i in range(X.shape[1])] + [('Y', f'Y[{i:d}]') for i in range(Y.shape[1])]
    df = DataFrame(concatenate((X, Y), axis=1), columns=MultiIndex.from_tuples(columns), dtype=float)
    return Store.from_df(folder=folder, df=df, meta=meta)


def ishigami(X: NP.MatrixLike, a: float = 7.0, b: float = 0.1) -> NP.Vector:
    """ The Ishigami function as described in https://www.sfu.ca/~ssurjano/ishigami.html.

    Args:
        X: The function argument, an (N,M) design Matrix. The standard distribution for each factor is ~U(-NP.pi, NP.pi).
        a: Parameter, normally equals 7.
        b: Parameter, normally equals 0.1 or 0.05.
    Returns: The (N,1) Vector ishigami(X).
    """
    X = atleast_2d(X)
    if 2 > X.shape[1]:
        return sin(X[:, [0]])
    elif 2 == X.shape[1]:
        return sin(X[:, [0]]) + a * sin(X[:, [1]]) ** 2
    else:
        # ishigami = (sin(X[:, [0]]) + (a * sin(X[:, [1]]) ** 2) + (b * (X[:, [2]] ** 4) * sin(X[:, [0]]))
        return (1 + b * X[:, [2]] ** 4) * sin(X[:, [0]]) + a * sin(X[:, [1]]) ** 2


def sobol_g(X: NP.MatrixLike, m_very_important: int = 0, m_important: int = 0, m_unimportant: int = 0,
            a_i_very_important: int = 0, a_i_important: int = 1, a_i_unimportant: int = 9) -> NP.Vector:
    """ The Sobol' G function as described in https://www.sfu.ca/~ssurjano/gfunc.html.

    Args:
        X: The function argument, an (N,M) design Matrix.

            The standard distribution for each factor is ~U(0, 1).
        m_very_important: Parameter, the number of better than important factors.
            Ignored if less than or equal to 0.
        m_important: Parameter, the number of better than unimportant factors.
            Ignored if less than or equal to m_very_important.
        m_unimportant: Parameter, the number of better than insignificant factors.
            Ignored if less than or equal to m_important, otherwise all factors beyond m_unimportant are insignificant.
        a_i_very_important: Parameter which conveys the importance of the ith factor
            for i &lt m_very_important. Defaults to 0 if m_very_important &gt 0, otherwise i/2.
        a_i_important: Parameter which conveys the importance of the i th factor for
            m_very_important &lt= i &lt m_important. Defaults to 1 if m_very_important &lt m_important, otherwise i/2.
        a_i_unimportant: Parameter which conveys the importance of the i th factor for
            m_important &lt= i &lt m_unimportant. Defaults to 9 if m_important &lt m_unimportant, otherwise i/2.
            The insignificant factors where m_unimportant &lt= i &lt M receive
            a_i_insignificant = i/2 whenever a_i_unimportant = i/2, otherwise 99.
    Returns: The (N,1) Vector g(X).
    """
    X = atleast_2d(X)

    def __calculate_a_i_times_2(M: int):
        """ Inner function to calculate a_i_times_2_inner."""
        a_i_times_2_inner = arange(0, M)  # Firstly, assume all a_i_ take the default value i/2
        #
        m_better_than_important = max(m_very_important, 0)
        a_i_times_2_inner[0: m_better_than_important] = a_i_very_important * 2
        #
        m_better_than_unimportant = max(m_important, m_better_than_important)
        a_i_times_2_inner[m_better_than_important: m_better_than_unimportant] = a_i_important * 2
        #
        if m_unimportant > m_better_than_unimportant:
            a_i_times_2_inner[m_better_than_unimportant: m_unimportant] = a_i_unimportant * 2
            a_i_times_2_inner[m_unimportant: M] = 198
        return a_i_times_2_inner

    a_i_times_2 = __calculate_a_i_times_2(X.shape[1])
    factors = (abs(8 * X - 4) + a_i_times_2) / (a_i_times_2 + 2)
    return prod(factors, axis=1, keepdims=True)


class FunctionWithParameters(Generic[NP.VectorOrMatrix]):
    """ A class for use with functions.sample(...). Encapsulates a function and its parameters."""

    @classmethod
    @property
    def DEFAULTS(cls) -> Dict[str, Tuple[float, float, Tuple[Callable[..., NP.VectorOrMatrix], Dict[str, Numeric]]]]:
        """ Default CDF_loc, CDF_scale and (function, parameters) indexed by function_name"""
        return {'sin.1': (pi, 2 * pi, (ishigami, {'a': 0.0, 'b': 0.0})),
                'sin.2': (pi, 2 * pi, (ishigami, {'a': 2.0, 'b': 0.0})),
                'ishigami': (pi, 2 * pi, (ishigami, {'a': 7.0, 'b': 0.1})),
                'sobol_g': (0.0, 1.0, (sobol_g, {'m_very_important': 0, 'm_important': 0, 'm_unimportant': 0,
                                                 'a_i_very_important': 0, 'a_i_important': 1, 'a_i_unimportant': 9})),
                'sobol_g_234': (0.0, 1.0, (sobol_g, {'m_very_important': 2, 'm_important': 3, 'm_unimportant': 4})),
                }

    @classmethod
    def default(cls, function_names: Sequence[str]) -> Tuple[NP.CovectorLike, NP.CovectorLike, Tuple[FunctionWithParameters, ...]]:
        """ Construct default CDF_loc, CDF_scale and functionsWithParameters for ``function_names``.

        Args:
            function_names: A Sequence of names of a Callable[..., NP.VectorOrMatrix].
        Returns: Tuples CDF_loc, CDF_scale and functionsWithParameters.

        Raises:
            KeyError: If one of the function_names is not recognized.
        """
        function_names = (function_names,) if isinstance(function_names, str) else function_names
        CDF_loc, CDF_scale, functions_with_parameters = zip(*(FunctionWithParameters.DEFAULTS[function_name] for function_name in function_names))
        return CDF_loc, CDF_scale, tuple(FunctionWithParameters(*fwp) for fwp in functions_with_parameters)

    @classmethod
    def from_meta(cls, meta: Dict[str, Any]) -> FunctionWithParameters:
        """ Construct a FunctionWithParameters from meta data.

        Args:
            meta: A ``dict`` produced by ``functionWithParameters.meta``.
        Returns: The FunctionWithParameters described by ``meta``.
        """
        _function = globals()[meta['functions']]
        _parameters = {key: (array(val) if isinstance(val, list) else val) for key, val in meta['parameters']}
        return cls(_function, _parameters)

    @property
    def function(self) -> Callable[..., NP.VectorOrMatrix]:
        """ The functions of this FunctionWithParameters."""
        return self._function

    def __call__(self, *args, **kwargs):
        kwargs = self._parameters | kwargs
        return self._function(*args, **kwargs)

    @property
    def parameters(self) -> Dict[str, Numeric]:
        """ The parameters of this FunctionWithParameters."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: Dict[str, Numeric]):
        self._parameters = value

    @property
    def meta(self) -> Dict[str, Any]:
        """ The meta data of this FunctionWithParameters."""
        parameters_meta = {key: (val.tolist() if isinstance(val, ndarray) else val) for key, val in self._parameters.items()}
        return {'functions': self._function.__name__, 'parameters': parameters_meta}

    def __init__(self, function_: Callable[..., NP.VectorOrMatrix], parameters_: Dict[str, Numeric]):
        """ Construct function_ with parameters_.

        Args:
            function_: The function of this FunctionWithParameters.
            parameters_: The parameters of this FunctionWithParameters.
        """
        self._function = function_
        self._parameters = parameters_


def functions_of_normal(store_dir: PathLike, N: int, M: int, CDF_loc: NP.CovectorLike = None, CDF_scale: NP.CovectorLike = None,
                        input_transform: FunctionWithParameters[NP.Matrix] = None, functions: Sequence[FunctionWithParameters[NP.Vector]] = None,
                        noise_std: NP.CovectorLike = 0.0) -> Store:
    """ Apply ``functions`` to ``N`` datapoints from ``input_transform(cdf(Multivariate.Normal(mean=zeros(M), covariance=eye(M))))`` then add noise.
        If CDF_loc is None or CDF_scale is None, the raw input X is used as the argument to input_transform.
        If input_transform is None the identity function is used instead. If functions is None, the identity functions is used instead.

    Args:
        store_dir: The location of the Store to be created for the results.
        N: The number of sample points.
        M: The number of input dimensions for the (N,M) design matrix.
        CDF_loc: An NP.CovectorLike of offsets to subtract from scaled CDFs.
            If None, CDFs are not used, else X is transformed into CDF_scale * CDF(X) - CDF_loc
        CDF_scale: An NP.CovectorLike of scalings for CDFs. If None, CDFs are not used, else X is transformed into CDF_scale * CDF(X) - CDF_loc.
        input_transform: This FunctionWithParameters transforms the (N,M) design Matrix produced according to X_distribution, X_sample_design
            and CDF_scale, CDF_loc to another (N,M) design matrix. Use this to combine inputs, to generate dependencies for example.
            Note that this functions must be of full rank -- it must not reduce M. If None the identity functions is assumed.
        functions: A list of FunctionWithParameters, each consisting of a functions(Callable) and its parameters(Dict).
            If None the identity functions is assumed.
        noise_std: An NP.CovectorLike to construct noise_distribution=Multivariate.Normal(mean=zeros(L), covariance=noise_std ** 2 * eye(L))

    Returns: data.Store.from_df(store_dir, df, meta). The calling arguments are encapsulated in meta, df consists of X and
        Y = functions(input_transform(CDF_scale * CDF(X) - CDF_loc)) + noise.
    """
    X_distribution = Multivariate.Independent(M, Univariate('norm', loc=0, scale=1))
    L = M if functions is None else len(functions)
    noise_distribution = Multivariate.Independent(L, Univariate('norm', loc=0, scale=noise_std)) if noise_std > EFFECTIVELY_ZERO else None
    return sample(store_dir, N, X_distribution, SampleDesign.LATIN_HYPERCUBE, CDF_loc, CDF_scale, input_transform, functions, noise_distribution,
                  SampleDesign.LATIN_HYPERCUBE)
