# BSD 3-Clause License
#
# Copyright (c) 2019-2021, Robert A. Milton
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" A suite of functions designed to test screening (order reduction) with high-dimensional distributions.

All functions herein are precisely as described in https://www.sfu.ca/~ssurjano/screen.html.
Each function signature follows the format:

|
``def function_(X: MatrixLike, **kwargs)``

    :X: The function argument, in the form of an ``NxM`` design Matrix.
    :**kwargs: Function-specific parameters, normally fixed.

Returns: A ``Vector[0 : N-1, 1]`` evaluating ``function_(X[0 : N-1, :])``.

|
"""

from numpy import atleast_2d, arange, prod, sin, einsum, full, concatenate, ndarray, array, eye
from pandas import DataFrame, MultiIndex
from romcomma.distribution import SampleDesign, Multivariate
from romcomma.typing_ import NP, Callable, Dict, Numeric, Sequence, NamedTuple, PathLike
from romcomma import data


def ishigami(X: NP.MatrixLike, a: float = 7.0, b: float = 0.1) -> NP.Vector:
    """ The Ishigami function as described in https://www.sfu.ca/~ssurjano/ishigami.html.

    Args:
        X: The function argument, an NxM design Matrix. The standard distribution for each factor is ~U(-NP.pi, NP.pi).
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
        return (1 + b * X[:, [2]] ** 4) * sin(X[:, [0]]) + a * sin(X[:, [1]]) ** 2


def sobol_g(X: NP.MatrixLike, m_very_important: int = 0, m_important: int = 0, m_unimportant: int = 0,
            a_i_very_important: int = 0, a_i_important: int = 1, a_i_unimportant: int = 9) -> NP.Vector:
    """ The Sobol' G function as described in https://www.sfu.ca/~ssurjano/gfunc.html.

    Args:
        X: The function argument, an NxM design Matrix.

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

    def _calculate_a_i_times_2(M: int):
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

    a_i_times_2 = _calculate_a_i_times_2(X.shape[1])
    factors = (abs(8 * X - 4) + a_i_times_2) / (a_i_times_2 + 2)
    return prod(factors, axis=1, keepdims=True)


def linear(X: NP.MatrixLike, matrix: NP.Matrix):
    """ Transform X by a square linear transformation matrix

    Args:
        X: An (N,M) design Matrix
        matrix: An (M,M) linear transformation matrix.
    Returns: An (N,M) design matrix given by the matrix product of X with matrix transpose.
    """
    return einsum('NO,MO -> NM', X, matrix, dtype=float)


def linear_matrix_from_meta(store: data.Store) -> NP.Matrix:
    """ Read linear transformation matrix used as pre_function_with_parameters in function.sample from the resulting __meta__.json
    Args:
        store: The data.Store

    Returns: The transformation matrix found, or the identity.

    """
    function_with_parameters = store.meta['origin']['functions_with_parameters'][0].split("; matrix=")
    if len(function_with_parameters) > 1:
        function_with_parameters = eval(function_with_parameters[-1][:-1])
        return array(function_with_parameters)
    else:
        return eye(store.meta['data']['M'], dtype=float)


class CallableWithParameters(NamedTuple):
    """ A NamedTuple for use with function.sample(...). Encapsulates a function and its parameters."""
    function: Callable[..., NP.Vector]
    parameters: Dict[str, Numeric]


def callable_with_parameters(function_: Callable[..., NP.Vector]) -> CallableWithParameters:
    """ Construct a suitable CallableWithParameters for ``function_``.

    Args:
        function_: The Callable of interest.
    Returns: A Tuple pairing function_ with a Dict of its default parameters.
        If function_ is not recognized, the Dict is left empty.
    """
    if function_.__name__ == "ishigami":
        parameters_ = {'a': 7.0, 'b': 0.1}
    elif function_.__name__ == "sobol_g":
        parameters_ = {'m_very_important': 0, 'm_important': 0, 'm_unimportant': 0, 'a_i_very_important': 0, 'a_i_important': 1, 'a_i_unimportant': 9}
    elif function_.__name__ == "linear":
        parameters_ = {'matrix': None}
    else:
        parameters_ = {}
    # noinspection PyTypeChecker
    return CallableWithParameters(function_, parameters_)


def sample(store_dir: PathLike, N: int, X_distribution: Multivariate, X_sample_design: SampleDesign = SampleDesign.LATIN_HYPERCUBE,
           CDF_scale: NP.ArrayLike = None, CDF_loc: NP.ArrayLike = None, pre_function_with_parameters: CallableWithParameters = None,
           functions_with_parameters: Sequence[CallableWithParameters] = None,
           noise_distribution: Multivariate = None, noise_sample_design: SampleDesign = SampleDesign.LATIN_HYPERCUBE) -> data.Store:
    """ Apply ``function_`` to ``N`` datapoints from ``X_distribution`` then add noise.

    Args:
        store_dir: The location of the Store to be created for the results.
        N: The number of sample points.
        X_distribution: The M-dimensional Multivariate distribution from which X is drawn.
        X_sample_design: SampleDesign.LATIN_HYPERCUBE or SampleDesign.RANDOM_VARIATE.
        CDF_scale: An NP.ArrayLike of scalings for CDFs. If None, CDFs are not used, else X is transformed into CDF_scale * CDF(X) - CDF_loc
        CDF_loc: An NP.ArrayLike of offsets to subtract from scaled CDFs.
        pre_function_with_parameters: A NamedTuple consisting of a function(Callable) and its parameters(Dict).
            This function which transforms the (N,M) design Matrix produced according to X_distribution, X_sample_design and CDF_scale
            to another (N,M) design matrix. Use this to combine inputs, to generate dependencies for example.
        functions_with_parameters: A list of NamedTuples, each consisting of a function(Callable) and its parameters(Dict).
            If None the identity function on X is assumed.
        noise_distribution: The Multivariate. distribution from which output noise is drawn, and added to the result of function_.
        noise_sample_design: SampleDesign.LATIN_HYPERCUBE or SampleDesign.RANDOM_VARIATE.
    Returns: data.Store.from_df(store_dir, df, meta). The calling arguments are encapsulated in meta, the result X: f(X) + noise in df.
    If functions_with_parameters is None  and pre_function_with_parameters is None then df represents X: X + noise.
    If noise_distribution is also None then df represents the sample X alone.

    Raises:
        IndexError: If noise.M is incommensurate with other arguments.
    """

    """ Validate arguments."""
    if not isinstance(X_distribution, Multivariate.Base):
        raise TypeError("type(X_distribution) is {t:s} when it must derive from Multivariate.Base."
                        .format(t=str(type(X_distribution))))
    if CDF_scale is not None:
        if not isinstance(X_distribution, Multivariate.Independent):
            raise TypeError("Can only CDFScale a distribution derived from Multivariate.Independent, not {t:s}.".format(t=str(type(X_distribution))))
        else:
            CDF_scale = atleast_2d(CDF_scale)
            if CDF_scale.shape == (1, 1):
                CDF_scale = full((1, X_distribution.M), CDF_scale[0, 0], dtype=float)
            if CDF_scale.shape not in ((1, X_distribution.M), (N, X_distribution.M)):
                raise IndexError("CDF_scale.shape={c}, X.shape={x}".format(c=CDF_scale.shape, x=(N, X_distribution.M)))
            CDF_loc = atleast_2d(0.0) if CDF_loc is None else atleast_2d(CDF_loc)
            if CDF_loc.shape == (1, 1):
                CDF_loc = full((1, X_distribution.M), CDF_loc[0, 0], dtype=float)
            if CDF_loc.shape not in ((1, X_distribution.M), (N, X_distribution.M)):
                raise IndexError("CDF_loc.shape={c}, X.shape={x}".format(c=CDF_loc.shape, x=(N, X_distribution.M)))
    if isinstance(functions_with_parameters, CallableWithParameters):
        functions_with_parameters = [functions_with_parameters]
    if noise_distribution is not None and not isinstance(noise_distribution, Multivariate.Base):
        raise TypeError("type(noise_distribution) is {t:s} when it must derive from Multivariate.Base.".format(t=str(type(noise_distribution))))

    X = X_distribution.sample(N, X_sample_design)
    columns = [("X", "[{i:d}]".format(i=i)) for i in range(X.shape[1])]
    meta = {'origin': {"N": N, "X_distribution": X_distribution.parameters, "CDF_scale": atleast_2d(CDF_scale).tolist(),
                       "CDF_loc": atleast_2d(CDF_loc).tolist(), "X_sample_design": str(X_sample_design), "functions_with_parameters": None,
                       "noise_distribution": None, "noise_sample_design": str(noise_sample_design)}}

    def _incorporated(_columns: list, _function_list: list):
        # noinspection PyTypeChecker
        meta['origin']["functions_with_parameters"] = _function_list
        _columns += list(zip(("Y",) * len(_function_list), _function_list))
        return DataFrame(concatenate((X, Y), axis=1), columns=MultiIndex.from_tuples(_columns), dtype=float), meta

    if functions_with_parameters is None:
        if CDF_scale is None:
            if pre_function_with_parameters is None:
                if noise_distribution is None:
                    return data.Store.from_df(dir_=store_dir, df=DataFrame(X, columns=MultiIndex.from_tuples(columns), dtype=float), meta=meta)
                else:
                    function_list = ["X[{i:d}]".format(i=i) for i in range(X.shape[1])]
                    Y = X.copy()
            else:
                function_list = [(pre_function_with_parameters.function.__name__ + "(X; "
                                  + ", ".join("{key}={val}".format(key=key, val=val) for (key, val) in pre_function_with_parameters.parameters.items())
                                  + ")[{i:d}]".format(i=i)) for i in range(X.shape[1])]
                Y = pre_function_with_parameters.function(X, **pre_function_with_parameters.parameters)
        else:
            if pre_function_with_parameters is None:
                function_list = ["{s:f} CDF(X[{i:d}]) - {l:f}".format(s=CDF_scale[0, i], l=CDF_loc[0, i], i=i) for i in range(X.shape[1])]
                Y = CDF_scale * X_distribution.cdf(X) - CDF_loc
            else:
                function_list = [("{f}({s} * CDF(X) - {l}; ".format(f=pre_function_with_parameters.function.__name__, s=CDF_scale.tolist(), l=CDF_loc.tolist())
                                  + ", ".join("{key}={val}".format(key=key, val=val.tolist() if isinstance(val, ndarray) else val)
                                              for (key, val) in pre_function_with_parameters.parameters.items()) + ")[{i:d}]".format(i=i)) for i in
                                 range(X.shape[1])]
                Y = CDF_scale * X_distribution.cdf(pre_function_with_parameters.function(X, **pre_function_with_parameters.parameters)) - CDF_loc
    else:
        if CDF_scale is None:
            if pre_function_with_parameters is None:
                function_list = [(f.function.__name__ + "(X; " + ", ".join("{key}={val}".format(key=key, val=val)
                                                                           for (key, val) in f.parameters.items()) + ")") for f in functions_with_parameters]
                Y = concatenate([f.function(X, **f.parameters) for f in functions_with_parameters], axis=1)
            else:
                function_list = [("{fn}({g}(X; ".format(fn=f.function.__name__, g=pre_function_with_parameters.function.__name__) +
                                  ", ".join("{key}={val}".format(key=key, val=val.tolist() if isinstance(val, ndarray) else val)
                                            for (key, val) in pre_function_with_parameters.parameters.items()) + ")") for f in functions_with_parameters]
                Y = concatenate([f.function(pre_function_with_parameters.function(X, **pre_function_with_parameters.parameters), **f.parameters)
                                 for f in functions_with_parameters], axis=1)
        else:
            if pre_function_with_parameters is None:
                function_list = [(f.function.__name__ + "({s} * CDF(X) - {l}; ".format(s=CDF_scale.tolist(), l=CDF_loc.tolist()) +
                                  ", ".join("{key}={val}".format(key=key, val=val)
                                            for (key, val) in f.parameters.items()) + ")") for f in functions_with_parameters]
                Y = concatenate([f.function(CDF_scale * X_distribution.cdf(X) - CDF_loc, **f.parameters) for f in functions_with_parameters], axis=1)
            else:
                function_list = [("{fn}({g}({s} * CDF(X) - {l}; ".format(fn=f.function.__name__, g=pre_function_with_parameters.function.__name__,
                                                                         s=CDF_scale.tolist(), l=CDF_loc.tolist()) +
                                  ", ".join("{key}={val}".format(key=key, val=val.tolist() if isinstance(val, ndarray) else val)
                                            for (key, val) in pre_function_with_parameters.parameters.items()) + ")") for f in functions_with_parameters]
                Y = concatenate([f.function(CDF_scale * X_distribution.cdf(pre_function_with_parameters.function(X, **pre_function_with_parameters.parameters))
                                            - CDF_loc, **f.parameters) for f in functions_with_parameters], axis=1)
    if noise_distribution is None:
        df, meta = _incorporated(columns, function_list)
        return data.Store.from_df(dir_=store_dir, df=df, meta=meta)
    if noise_distribution.M != Y.shape[1]:
        raise IndexError("noise.M = {n_m:d} != {l:d} = L = dimension output."
                         .format(n_m=noise_distribution.M, l=Y.shape[1]))
    Y += noise_distribution.sample(N, noise_sample_design)
    meta['origin']["noise_distribution"] = noise_distribution.parameters
    df, meta = _incorporated(columns, function_list)
    return data.Store.from_df(dir_=store_dir, df=df, meta=meta)
