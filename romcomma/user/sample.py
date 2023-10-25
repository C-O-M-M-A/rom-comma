#  BSD 3-Clause License.
#
#  Copyright (c) 2019-2023 Robert A. Milton. All rights reserved.
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

""" **Functionality for sampling and Design of Experiments (DOE)** """

from __future__ import annotations

import numpy as np

from romcomma.base.definitions import *
import scipy.stats
from romcomma.data.storage import Frame, Repository, Fold
from romcomma.user import functions
import shutil
import sys


def permute_axes(new_order: Sequence | None) -> NP.Matrix | None:
    """ Provide a rotation matrix which reorders axes. Most use cases are to re-order input axes according to GSA.

    Args:
        new_order: A Tuple or List containing a permutation of ``[0,...,M-1]``, for passing to ``np.transpose``.

    Returns: A rotation matrix which will reorder the axes to new_order. Returns ``None`` if ``new_order is None``.
    """
    return None if new_order is None else np.eye(len(new_order))[new_order, :]


class DOE:
    """ Sampling methods for inputs."""

    Method: Type = Callable[[int, int, Any], NP.Matrix]     #: Function signature of a DOE method.

    @staticmethod
    def latin_hypercube(N: int, M: int, is_centered: bool = True):
        """ Latin Hypercube DOE.

        Args:
            N: The number of samples (rows).
            M: The of input dimensions (columns).
            is_centered: Boolean ordinate whether to centre each sample in its Latin Hypercube cell.
                Default is False, which locates the sample randomly within its cell.
        Returns: An (N,M) matrix of N samples of dimension M.
        """
        return scipy.stats.qmc.LatinHypercube(M, scramble=not is_centered).random(N)

    @staticmethod
    def full_factorial(N: int, M: int):
        """ Full factorial DOE.

        Args:
            N: The number of samples (rows).
            M: The of input dimensions (columns).
        Returns: An (N,M) matrix of N samples of dimension M.
        """
        NM = N // M
        N1 = N - M * NM
        return np.concatenate([1/(2 * N1) + np.linspace(0, 1, N1, False),] + (M-1) * [1/(2 * NM) + np.linspace(0, 1, NM, False), ], axis=1)


class GaussianNoise:
    """ Sample multivariate, zero-mean Gaussian noise. """

    class Variance:
        """ An artificially generated (co)variance matrix for GaussianNoise, with a useful labelling scheme."""

        @property
        def matrix(self) -> NP.Matrix:
            """ Variance as an (L,L) covariance matrix, suitable for constructing GaussianNoise."""
            return self._matrix

        @property
        def meta(self) -> Dict[str, Any]:
            """ Meta data for providing to ``data.storage``. This matches the initials in ``self.__format__()``."""
            return {'generator': 'determined' if self.is_determined else 'undetermined',
                    'is_covariant': 'covariance' if self.is_covariant else 'variance', 'magnitude': self.magnitude}

        def __call__(self) -> NP.Matrix:
            """
            Returns: Variance as an (L,L) covariance matrix, suitable for constructing GaussianNoise.
            The constructor generates the matrix (perhaps stochastically), so repeated calls to any methods produce identical Variance.
            """
            return self.matrix

        def __format__(self, format_spec: Any) -> str:
            """ The label for this Variance, to help name samples informatively. The description is ``d.`` (determined) or ``u.`` (undetermined),
            followed by ``v.`` (diagonal variance) or ``c.`` (non-diagonal covariance), followed by ``100 * self.magnitude:.2f``."""
            return f'{"d." if self.is_determined else "u."}{"c." if self.is_covariant else "v."}{100 * self.magnitude:.2f}'

        def __init__(self, L: int, magnitude: float, is_covariant: bool = False, is_determined: bool = True):
            """ Instantiate an (L,L) GaussianNoise (co)variance matrix.

            Args:
                L: Output dimensionality.
                magnitude: The StdDev of noise.
                is_covariant: True to create a diagonal variance matrix.
                is_determined: False to create a random symmetric matrix.
            """
            self.magnitude, self.is_covariant, self.is_determined = magnitude, is_covariant, is_determined
            if self.is_determined:
                self._matrix = 2 * np.random.random_sample((L, L)) - np.ones((L, L))
                self._matrix = np.matmul(self._matrix, self._matrix.transpose())
                self._matrix /= np.trace(self._matrix) / L
            else:
                self._matrix = np.array([[(-1)**(i-j)/(1.0 + abs(i-j)) for i in range(L)] for j in range(L)])
            if not self.is_covariant:
                self._matrix = np.diag(np.diag(self._matrix))
            self._matrix *= self.magnitude ** 2

    @property
    def variance(self) -> NP.Matrix:
        return self._variance

    def __call__(self, repo: Repository | None = None) -> NP.Matrix:
        """ Generate N samples of L-dimensional Gaussian noise, sampled from :math:`N[0,self.variance]`.
        The constructor generates the sample, so repeated calls to any method always refer to the same GaussianNoise.

        Args:
            repo: An optional Repository which will have GaussianNoise added to Y in data.csv.
        Returns: An (N,L) noise matrix, where (L,L) is the shape of `self._variance`.
        """

        if repo is not None:
            repo.data.df.iloc[:, :] = np.concatenate((repo.X, repo.Y + self._rvs), axis=1)
            repo.data.write()
        return self._rvs

    def __init__(self, N: int, variance: NP.MatrixLike):
        """ Generate N samples of L-dimensional Gaussian noise, sampled from :math:`\mathsf{N}[0,variance]`.

        Args:
            N: Number of samples (rows).
            variance: (L,L) covariance matrix for homoskedastic noise.
        """
        self._variance = np.atleast_2d(variance)
        if len(self._variance.shape) == 2 and self._variance.shape[0] == 1:
            self._variance = np.diagflat(self._variance)
        elif self._variance.shape[0] != self._variance.shape[1] or len(self._variance.shape) > 2:
            raise IndexError(f'variance.shape = {self._variance.shape} should be (L,) or (L,L).')
        self._rvs = scipy.stats.multivariate_normal.rvs(mean=None, cov=self._variance, size=N)
        self._rvs.shape = (N, self._variance.shape[1])


class Function:
    """ Sample a ``user.function.Vector``."""

    @property
    def repo(self) -> Repository:
        """ The Repository containing the Function sample."""
        return self._repo

    def collection(self, sub_folder: Union[Path, str]) -> Dict[str, Any]:
        """ Construct a Dict for user.results.Collect, with appropriate ``extra_columns``.

        Args:
            folder: The folder under ``self.repo.folder`` housing the csvs to collect.
        Returns: The Dict for ``self.repo``.
        """
        return {'folder': self._repo.folder / sub_folder, 'N': self._N, 'noise': self._noise_variance.magnitude}

    def un_rotate_folds(self) -> Function:
        """ Create an un-rotated Fold in the Repository, with index ``K+1``."""
        shutil.copytree(self._repo.fold_folder(self._repo.K), self._repo.fold_folder(self._repo.K + 1))
        fold = Fold(self._repo, self._repo.K + 1)
        fold.X_rotation = np.transpose(fold.X_rotation)
        Frame(fold.test_csv, fold.normalization.undo_from(fold.test_data.df))
        fold = Fold(self._repo, self._repo.K)
        Frame(self._repo.folder / 'undo_from.csv', fold.normalization.undo_from(fold.test_data.df))
        return self

    def _construct(self, folder: Path | str, X: NP.Matrix, function_vector: functions.Vector, noise: NP.Matrix, origin_meta: Dict[str, Any]) -> Repository:
        """ Construct Repository housing the sample design matrix ``(X, f(X) + noise)``.

        Args:
            folder: The Repository folder.
            X: An (N,M) design matrix of inputs.
            function_vector: An (L,) function.Vector.
            noise: An (N,L) design matrix of noise.
            origin_meta: A Dict of meta specifying the origin of the sample.
        Returns: The ``(X, f(X) + noise)`` sample design matrix Repository, before folding or rotating.
        """
        Y = function_vector(X)
        std = np.reshape(np.std(Y, axis=0), (1, -1))
        Y += std * noise
        columns = [('X', f'X.{i:d}') for i in range(X.shape[1])] + [('Y', f'Y.{i:d}') for i in range(Y.shape[1])]
        df = pd.DataFrame(np.concatenate((X, Y), axis=1), columns=pd.MultiIndex.from_tuples(columns), dtype=float)
        return Repository.from_df(folder=folder, df=df, meta={'origin': origin_meta})

    def __init__(self, root: Path | str, doe: DOE.Method, function_vector: functions.Vector, N: int, M: int, noise_variance: GaussianNoise.Variance,
                 ext: str | None = None, overwrite_existing: bool = False, **kwargs: Any):
        """ Construct a Repository by sampling a function over a DOE.

        Args:
            root: The folder under which the Repository will sit.
            doe: An experimental design for the sample inputs.
            function_vector: A vector function.
            N: The number of samples (rows) in the sample.
            M: The input dimensionality (columns).
            noise_magnitude: The (L,L) homoskedastic ``GaussianNoise.Variance``.
            ext: Unless None, the repo name is suffixed by ``.[ext]``.
            overwrite_existing: Whether to overwrite an existing Repository.
            **kwargs: Options passed straight to doe.
        """
        self._N, self._noise_variance = N, noise_variance
        folder = Path(root) / f'{function_vector.name}.M.{M:d}.{self._noise_variance}.N.{N:d}{"" if ext is None else "." + ext}'
        if folder.is_dir() and not overwrite_existing:
            self._repo = Repository(folder)
        else:
            self._repo = self._construct(folder=folder, X=doe(N, M, **kwargs), function_vector=function_vector,
                                         noise=GaussianNoise(N, self._noise_variance())(repo=None),
                                         origin_meta={'DOE': doe.__name__, 'function_vector': function_vector.meta, 'noise': self._noise_variance.meta})
            Frame(folder / 'likelihood.variance.csv', pd.DataFrame(self._noise_variance()))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Please provide at least 3 arguments: The folder to write to, the number of input dimensions (columns) M, and at least one value for the number '
              'of samples (rows) N.')
    else:
        root = Path(sys.argv[1])
        M = int(sys.argv[2])
        for N in sys.argv[3:]:
            N = int(N)
            pd.DataFrame(DOE.latin_hypercube(N, M)).to_csv(root / f'lhs.{N}.csv')

