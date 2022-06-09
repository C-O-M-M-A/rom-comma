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

""" Contains a suite of functions designed to test_data screening (order reduction) with high-dimensional distributions.

All functions herein are taken from https://salib.readthedocs.io.
Each function signature follows the format:

|
``def function_(X: MatrixLike, **kwargs)``

    :X: The function argument, in the form of an ``(N,M)`` design Matrix.
    :**kwargs: Function-specific parameters, normally fixed.

Returns: A ``Vector[0 : N-1, 1]`` evaluating ``function_(X[0 : N-1, :])``.

|
"""

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.data.storage import Repository
from romcomma.test import sampling
from SALib.test_functions import Ishigami, Sobol_G


class FunctionWithMeta:
    """ A class for use with functions.sample(...). Encapsulates a function and its parameters."""

    _DEFAULT = None

    @classmethod
    def _default(cls, **kwargs: Any) -> Dict[str, FunctionWithMeta]:
        name = kwargs['name']
        return {name: FunctionWithMeta(**kwargs)}

    @classmethod
    @property
    def DEFAULT(cls) -> Dict[str, FunctionWithMeta]:
        """ List of Default FunctionsWithMeta."""
        if cls._DEFAULT is None:
            cls._DEFAULT = {
                **cls._default(name='sin.1', function=Ishigami.evaluate, loc=-np.pi, scale=2 * np.pi, A=0.0, B=0.0),
                **cls._default(name='sin.2', function=Ishigami.evaluate, loc=-np.pi, scale=2 * np.pi, A=2.0, B=0.0),
                **cls._default(name='ishigami', function=Ishigami.evaluate, loc=-np.pi, scale=2 * np.pi, A=7.0, B=0.1),
                **cls._default(name='sobol_g', function=Sobol_G.evaluate, loc=0, scale=1, a=np.array([0, 1, 4.5, 9, 99])),
                **cls._default(name='sobol_g2', function=Sobol_G.evaluate, loc=0, scale=1, a=np.array([0, 1, 4.5, 9, 99]),
                               alpha=np.array([2.0, 2.0, 2.0, 2.0, 2.0])),
            }
        return cls._DEFAULT

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta | {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in self.parameters.items()}

    def __call__(self, X: NP.Matrix, **kwargs) -> NP.Matrix:
        kwargs.update(self.parameters)
        return np.reshape(self._function(X * self._meta['scale'] + self._meta['loc'], **kwargs), (X.shape[0], 1))

    def __init__(self, **kwargs):
        self._meta = {key: kwargs.pop(key) for key in ('name', 'loc', 'scale')}
        self._function = kwargs.pop('function')
        self.parameters = kwargs.copy()


def sample(functions: Tuple[FunctionWithMeta], N: int, M: int, likelihood_variance: NP.MatrixLike, folder: PathLike,
           sampling_method: Callable[[int, int, Any], NP.Matrix] = sampling.latin_hypercube, **kwargs) -> Repository:
    """ Record a sample of test function responses.

    Args:
        functions: A tuple of test functions, of length L.
        N: The number of samples (datapoints), N &gt 0.
        M: The input dimensionality, M &ge 0.
        likelihood_variance: A noise (co)variance of shape (L,L) or (L,). The latter is interpreted as an (L,L) diagonal matrix.
            Used to generate N random samples of Gaussian noise ~ N(0, noise_variance).
        folder: The Repository.folder to create and record the results in.
        sampling_method: A Callable sampling_method(N, M, **kwargs) -> X, which returns an (N,M) matrix.
        kwargs: Passed directly to sampling_method.
    Returns: A Repository containing N rows of M input columns and L output columns. The output is f(X) + noise.
    """
    X = sampling_method(N, M, **kwargs)
    likelihood_variance = np.atleast_2d(likelihood_variance)
    origin_meta = {'sampling_method': sampling_method.__name__, 'noise_variance': likelihood_variance.tolist()}
    noise = sampling.multivariate_gaussian_noise(N, likelihood_variance)
    return apply(functions, X, noise, folder, origin_meta=origin_meta)


def apply(functions: Tuple[FunctionWithMeta], X: NP.Matrix, noise: NP.Matrix, folder: PathLike, **kwargs) -> Repository:
    """ Record a sample of test function responses.

    Args:
        functions: A tuple of test functions, of length L.
        X: An (N,M) design matrix of inputs.
        noise: An (N,L) design matrix of fractional output noise per unit Y.
        folder: The Repository.folder to create and record the results in.
    Returns: A repo containing N rows of M input columns and L output columns. The output is f(X) + noise.
    Raises: IndexError if dimensions are incompatible.
    """
    X, noise = np.atleast_2d(X), np.atleast_2d(noise)
    if min(X.shape) < 1:
        raise IndexError(f'X.shape = {X.shape} does not consist of two non-zero dimensions.')
    if min(noise.shape) < 1:
        raise IndexError(f'noise.shape = {noise.shape} does not consist of two non-zero dimensions.')
    if X.shape[0] != noise.shape[0]:
        raise IndexError(f'X has {X.shape[0]} samples while noise has {noise.shape[0]} samples.')
    if len(functions) == 1:
        functions = functions * noise.shape[1]
    elif len(functions) != noise.shape[1]:
        raise IndexError(f'functions should be of length L, equal to noise.shape[1].')
    meta = {'N': X.shape[0], 'functions': [f.meta for f in functions]}
    meta = {'origin': meta | kwargs.get('origin_meta', {})}
    Y = np.concatenate([f(X) for f in functions], axis=1)
    std = np.reshape(np.std(Y, axis=0), (1, -1))
    Y += noise * std
    columns = [('X', f'X.{i:d}') for i in range(X.shape[1])] + [('Y', f'Y.{i:d}') for i in range(Y.shape[1])]
    df = pd.DataFrame(np.concatenate((X, Y), axis=1), columns=pd.MultiIndex.from_tuples(columns), dtype=float)
    return Repository.from_df(folder=folder, df=df, meta=meta)
