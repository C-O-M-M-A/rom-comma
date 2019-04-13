""" A suite of functions designed to test screening (order reduction) with high-dimensional distributions.

All functions herein are precisely as described in https://www.sfu.ca/~ssurjano/screen.html.
Each function signature follows the format:

|
``def function_(X: MatrixLike, **kwargs)``

:X: The function argument, in the form of an ``NxM`` design matrix.
:**kwargs: Function-specific parameters, normally fixed.

Returns: A ``CoVector[0 : N-1, 1]`` evaluating ``function_(X[0 : N-1, :])``.

|
Contents:
    :ishigami: function.
    :sobol_g: function.
    :CallableWithParameters: type(``Tuple[Callable[..., NP.Covector], Dict[str, Numeric]]``).
    :callable_with_parameters: function.
    :sample: function. Umbrella for generating samples.
"""

from numpy import atleast_2d, arange, hstack, prod, sin
from pandas import DataFrame, MultiIndex
from romcomma.distribution import SampleDesign, Multivariate
from romcomma.typing_ import NP, Callable, Dict, Numeric, Tuple, Sequence, Union, Type


# noinspection PyPep8Naming
def ishigami(X: NP.MatrixLike, pre_function: Callable[[NP.MatrixLike], NP.MatrixLike] = None, a: float = 7.0, b: float = 0.1) -> NP.Covector:
    """ The Ishigami function as described in https://www.sfu.ca/~ssurjano/ishigami.html.

    Args:
        X: The function argument, an ``NxM`` design matrix. The standard distribution for each factor is ~U(-``NP.pi``, ``NP.pi``).
        pre_function: A function applied to X prior to ishigami.
        a: Parameter, normally equals ``7``.
        b: Parameter, normally equals ``0.1`` or ``0.05``.

    Returns: The Covector ``ishigami(X[0 : N-1, :])`` of length ``N``.
    """
    X = atleast_2d(X)
    if pre_function is not None:
        X = atleast_2d(pre_function(X))
    if 2 > X.shape[1]:
        return sin(X[:, 0:1])
    elif 2 == X.shape[1]:
        return sin(X[:, 0:1]) + a * sin(X[:, 1:2]) ** 2
    else:
        return (1 + b * X[:, 2:3] ** 4) * sin(X[:, 0:1]) + a * sin(X[:, 1:2]) ** 2


# noinspection PyPep8Naming
def sobol_g(X: NP.MatrixLike, m_very_important: int = 0, m_important: int = 0, m_unimportant: int = 0,
            a_i_very_important: int = 0, a_i_important: int = 1, a_i_unimportant: int = 9) -> NP.Covector:
    """ The Sobol' G function as described in https://www.sfu.ca/~ssurjano/gfunc.html.

    Args:
        X: The function argument, an ``NxM`` design matrix.

            The standard distribution for each factor is ~U(``0``, ``1``).
        m_very_important: Parameter, the number of better than important factors.
            Ignored if less than or equal to ``0``.
        m_important: Parameter, the number of better than unimportant factors.
            Ignored if less than or equal to ``m_very_important``.
        m_unimportant: Parameter, the number of better than insignificant factors.
            Ignored if less than or equal to ``m_important``, otherwise all factors beyond ``m_unimportant`` are insignificant.
        a_i_very_important: Parameter which conveys the importance of the ``i`` th factor
            for ``i < m_very_important``. Defaults to ``0`` if ``m_very_important > 0``, otherwise ``i/2``.
        a_i_important: Parameter which conveys the importance of the ``i`` th factor for
            ``m_very_important <= i < m_important``. Defaults to ``1`` if ``m_very_important`` < ``m_important``, otherwise ``i/2``.
        a_i_unimportant: Parameter which conveys the importance of the ``i`` th factor for
            ``m_important <= i < m_unimportant``. Defaults to ``9`` if ``m_important`` < ``m_unimportant``, otherwise ``i/2``.
            The insignificant factors where ``m_unimportant <= i < M`` receive
            ``a_i_insignificant = i/2`` whenever ``a_i_unimportant = i/2``, otherwise ``99``.

    Returns: The Covector ``sobol_g(X[0 : N-1, :])`` of length ``N``.
    """
    X = atleast_2d(X)

    # noinspection PyPep8Naming
    def _calculate_a_i_times_2(M: int):
        """ Inner function to calculate a_i_times_2_inner"""
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


CallableWithParameters: Type = Tuple[Callable[..., NP.Covector], Dict[str, Numeric]]


def callable_with_parameters(function_: Callable[..., NP.Covector]) -> CallableWithParameters:
    """ Construct a suitable ``function.CallableWithParameters`` for ``function_``.

    Args:
        function_: The ``Callable`` of interest.
    Returns: A ``tuple`` pairing ``function_`` with a ``dict`` of its default parameters.
        If ``function_`` is not recognized, the ``dict`` is left empty.
    """
    if function_.__name__ == "ishigami":
        parameters_ = {'a': 7.0, 'b': 0.1}
    elif function_.__name__ == "sobol_g":
        parameters_ = {'m_very_important': 0, 'm_important': 0, 'm_unimportant': 0, 'a_i_very_important': 0, 'a_i_important': 1, 'a_i_unimportant': 9}
    else:
        parameters_ = {}
    return function_, parameters_


# noinspection PyPep8Naming
def sample(N: int, X_distribution: Union[Multivariate.Independent, Multivariate.Normal], CDF_Scale: NP.ArrayLike = None,
           X_generator: SampleDesign = SampleDesign.LATIN_HYPERCUBE,
           functions_with_parameters: Sequence[CallableWithParameters] = None,
           noise_distribution: Union[Multivariate.Independent, Multivariate.Normal] = None,
           noise_generator: SampleDesign = SampleDesign.LATIN_HYPERCUBE) -> Tuple[DataFrame, dict]:
    """ Apply ``function_`` to ``N`` from ``X_distribution`` then add noise.

    Args:
        N: The number of sample points.
        X_distribution: The ``Multivariate.Independent`` distribution from which ``X`` is drawn.
        CDF_Scale: An NP.ArrayLike of scalings for CDFs. If None, CDFs are not used, else
        X_generator: ``SampleDesign.LATIN_HYPERCUBE`` or
            ``SampleDesign.RANDOM_VARIATE``.
        functions_with_parameters: A list of tuple pairs, each consisting of a function(Callable)
            and its parameters(Dict). If ``None`` the identity function on ``X`` is assumed.
        noise_distribution: The Multivariate. distribution from which output noise is drawn, and added to the result of
            ``function_``.
        noise_generator: ``SampleDesign.LATIN_HYPERCUBE`` or
            ``SampleDesign.RANDOM_VARIATE``.
    Returns: ``data(DataFrame), meta(dict)``. The calling arguments are encapsulated in ``meta``, the result
        ``X: f(X) + noise`` in ``data``.
        If ``functions_with_parameters is None`` then ``data`` represents ``X: X + noise``.
        If ``noise_distribution is also ``None`` then ``data`` represents the sample ``X`` alone.
    Raises:
        IndexError: If noise.M is incommensurate with other arguments.
    """
    if not isinstance(X_distribution, Multivariate.Base):
        raise TypeError("type(X_distribution) is {t:s} when it must derive from Multivariate.Base."
                        .format(t=str(type(X_distribution))))
    if CDF_Scale is not None:
        if not isinstance(X_distribution, Multivariate.Independent):
            raise TypeError("Can only CDFScale a distribution derived from Multivariate.Independent, not {t:s}."
                            .format(t=str(type(X_distribution))))
        else:
            CDF_Scale = atleast_2d(CDF_Scale)
            if CDF_Scale.shape not in ((1,1),(1,X_distribution.M),(N, X_distribution.M)):
                raise IndexError("CDF_Scale.shape={c}, X.shape={x}".format(c=CDF_Scale.shape, x=(N, X_distribution.M)))

    if noise_distribution is not None and not isinstance(noise_distribution, Multivariate.Base):
        raise TypeError("type(noise_distribution) is {t:s} when it must derive from Multivariate.Base."
                        .format(t=str(type(noise_distribution))))
    X = X_distribution.sample(N, X_generator)
    columns = [("X", "[{i:d}]".format(i=i)) for i in range(X.shape[1])]
    meta = {'origin': {"N": N, "X_distribution": X_distribution.parameters, "CDF_Scale": CDF_Scale, "X_generator": str(X_generator),
                       "functions_with_parameters": None, "noise_distribution": None, "noise_generator": str(noise_generator)}}

    def _incorporated(_columns: list, _function_list: list):
        # noinspection PyTypeChecker
        meta['origin']["functions_with_parameters"] = _function_list
        _columns += list(zip(("Y",) * len(_function_list), _function_list))
        return DataFrame(hstack((X, Y)), columns=MultiIndex.from_tuples(_columns), dtype=float), meta

    if functions_with_parameters is None:
        if noise_distribution is None:
            return DataFrame(X, columns=MultiIndex.from_tuples(columns), dtype=float), meta
        else:
            if noise_distribution.M != X_distribution.M:
                raise IndexError("noise.M = {n_m:d} != {f_m:d} = X.M = dimension of identity function."
                                 .format(n_m=noise_distribution.M, f_m=X_distribution.M))
            if CDF_Scale is None:
                function_list = ["X[{i:d}]".format(i=i) for i in range(X.shape[1])]
                Y = X.copy()
            else:
                function_list = ["{s:f} CDF(X[{i:d})]".format(s=CDF_Scale[0, i], i=i) for i in range(X.shape[1])]
                Y = CDF_Scale * X_distribution.cdf(X)
    else:
        if CDF_Scale is None:
            function_list = [(function_.__name__ + "(X; " + ", ".join("{key}={val}".format(key=key, val=val)
                                                                      for (key, val) in parameters_.items()) + ")")
                             for function_, parameters_ in functions_with_parameters]
            Y = hstack([function_(X, **parameters_) for function_, parameters_ in functions_with_parameters])
        else:
            function_list = [(function_.__name__ + "({s:f} CDF(X); ".format(s=CDF_Scale) + ", ".join("{key}={val}".format(key=key, val=val)
                                                                      for (key, val) in parameters_.items()) + ")")
                             for function_, parameters_ in functions_with_parameters]
            X_cdf = CDF_Scale * X_distribution.cdf(X)
            Y = hstack([function_(X_cdf, **parameters_) for function_, parameters_ in functions_with_parameters])
        if noise_distribution is None:
            return _incorporated(columns, function_list)
        else:
            if noise_distribution.M != len(functions_with_parameters):
                raise IndexError("noise.M = {n_m:d} != {f_m:d} = Number of functions = Y.M."
                                 .format(n_m=noise_distribution.M, f_m=len(functions_with_parameters)))
    Y += noise_distribution.sample(N, noise_generator)
    meta['origin']["noise_distribution"] = noise_distribution.parameters
    return _incorporated(columns, function_list)
