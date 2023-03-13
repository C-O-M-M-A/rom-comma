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

# Contains Sampling and Design of Experiments functionality.

from __future__ import annotations

from romcomma.base.definitions import *
import scipy.stats
from romcomma.data.storage import Frame, Repository, Fold
import romcomma.test.functions as functions
import shutil
import sys


class DOE:
    """ Sampling methods for inputs."""

    Method: Type = Callable[[int, int, Any], NP.Matrix]

    @staticmethod
    def latin_hypercube(N: int, M: int, is_centered: bool = True):
        """ Latin Hypercube Sample.

        Args:
            N: The number of samples (datapoints).
            M: The dimensionality of the hypercube.
            is_centered: Boolean ordinate whether to centre each sample in its Latin Hypercube cell.
                Default is False, which locates the sample randomly within its cell.
        Returns: An (N,M) matrix of N datapoints of dimension M.
        """
        return scipy.stats.qmc.LatinHypercube(M, centered=is_centered).random(N)

    @staticmethod
    def full_factorial(N: int, M: int):
        """ Full factorial Sample.

        Args:
            N: The number of samples (datapoints).
            M: The dimensionality of the hypercube.
        Returns: An (N,M) matrix of N datapoints of dimension M.
        """
        NM = N // M
        N1 = N - M * NM
        return np.concatenate([1/(2 * N1) + np.linspace(0, 1, N1, False),] + (M-1) * [1/(2 * NM) + np.linspace(0, 1, NM, False),], axis=1)


class GaussianNoise:
    """ Multivariate, zero-mean Gaussian noise. """

    class Variance:
        """ An artificially generated covariance matrix for GaussianNoise, with a useful labelling scheme."""

        @property
        def matrix(self) -> NP.Matrix:
            return self._matrix

        @property
        def meta(self) -> Dict[str, Any]:
            return {'variance': 'stochastic' if self.is_stochastic else 'fixed',
                    'structure': 'independent' if self.is_diagonal else 'dependent', 'magnitude': self.magnitude}

        def __call__(self) -> NP.Matrix:
            """
            Returns: Variance as an (L,L) covariance matrix.
            """
            return self.matrix

        def __format__(self, format_spec: Any) -> str:
            """ The label for this Variance, used to name samples."""
            return f'{"s." if self.is_stochastic else "f."}{"i." if self.is_diagonal else "d."}{100 * self.magnitude:.2f}'

        def __init__(self, L: int, magnitude: float, is_diagonal: bool = False, is_stochastic: bool = False):
            """ Instantiate an (L,L) GaussianNoise covariance matrix.

            Args:
                L: Output dimension.
                magnitude: The StdDev of noise.
                is_diagonal: True to create a diagonal variance matrix.
                is_stochastic: True to create a random matrix.
            """
            self.magnitude, self.is_diagonal, self.is_stochastic = magnitude, is_diagonal, is_stochastic
            if self.is_stochastic:
                self._matrix = 2 * np.random.random_sample((L, L)) - np.ones((L, L))
                self._matrix = np.matmul(self._matrix, self._matrix.transpose())
                self._matrix /= np.trace(self._matrix) / L
            else:
                self._matrix = np.array([[(-1)**(i-j)/(1.0 + abs(i-j)) for i in range(L)] for j in range(L)])
            if self.is_diagonal:
                self._matrix = self.magnitude * self.magnitude * np.diag(np.diag(self._matrix))
            self._matrix *= self.magnitude ** 2

    @property
    def variance(self) -> NP.Matrix:
        return self._variance

    def __call__(self, repo: Optional[Repository] = None) -> NP.Matrix:
        """ Generate N datapoints of L-dimensional Gaussian noise, sampled from N[0, self.variance].

        Args:
            repo: An optional Repository which will have GaussianNoise added to Y in data.csv.
        Returns: An (N,L) noise matrix, where (L,L) is the shape of `self._variance`, or the repo that this noise has been added to.
        """

        if repo is not None:
            result = np.concatenate((repo.X, repo.Y + self._rvs), axis=1)
            repo.data.df.iloc[:, :] = result
            repo.data.write()
        return self._rvs

    def __init__(self, N: int, variance: NP.MatrixLike):
        """ Generate an (N,L) sample of GaussianNoise.

        Args:
            N: Number of samples (datapoints).
            variance: (L,L) Covariance matrix for homoskedastic noise.
        """
        self._variance = np.atleast_2d(variance)
        if self._variance.shape[0] == 1 and len(self._variance.shape) == 2:
            self._variance = np.diagflat(self._variance)
        elif self._variance.shape[0] != self._variance.shape[1] or len(self._variance.shape) > 2:
            raise IndexError(f'variance.shape = {self._variance.shape} should be (L,) or (L,L).')
        self._rvs = scipy.stats.multivariate_normal.rvs(mean=None, cov=self._variance, size=N)
        self._rvs.shape = (N, self._variance.shape[1])


class Function:
    """ Sample a Vector of test functions."""

    @property
    def repo(self) -> Repository:
        """Returns: The Repository containing this sample."""
        return self._repo

    def aggregator(self, child_folder: Union[Path, str]) -> Dict[str, Any]:
        """ Construct an aggregator Dict.

        Args:
            child_folder: The child folder within the repo where the .csv to aggregate sits. The aggregate will sit under  base_folder/child_folder
        Returns: The aggregator for self._repo.
        """
        return {'folder': self._repo.folder / child_folder, 'N': self._N, 'noise': self._noise_variance.magnitude}

    def into_K_folds(self, K: int, shuffle_before_folding: bool = False, normalization: Optional[PathLike] = None) -> Function:
        """ Fold repository into K Folds, indexed by range(K).

        Args:
            K: The number of Folds, of absolute value between 1 and N inclusive.
                An improper Fold, indexed by K and including all data for both training and testing is included by default.
                To suppress this give K as a negative integer.
            shuffle_before_folding: Whether to shuffle the data before sampling.
            normalization: An optional normalization.csv file to use.

        Raises:
            IndexError: Unless 1 &lt= K &lt= N.
        """
        self._repo.into_K_folds(K, shuffle_before_folding, normalization)
        return self

    def random_rotation(self) -> NP.Matrix:
        """Returns: An (M,M) random rotation matrix. """
        return scipy.stats.ortho_group.rvs(self._repo.M)

    def rotate_folds(self, rotation: Optional[NP.Matrix]) -> Function:
        """ Uniformly rotate the folds in a Repository. The rotation (like normalization) applies to each fold, not the repo itself.

        Args:
            rotation: The (M,M) rotation matrix to apply to the inputs. If None, the identity matrix is used.
        """
        rotation = np.eye(self._repo.M)if rotation is None else rotation
        for k in self._repo.folds:
            Fold(self._repo, k).X_rotation = rotation
        return self

    def un_rotate_folds(self) -> Function:
        """ Create an Un-rotated Fold in the Repository, with index ``K+1``."""
        shutil.copytree(self._repo.fold_folder(self._repo.K), self._repo.fold_folder(self._repo.K + 1))
        fold = Fold(self._repo, self._repo.K + 1)
        fold.X_rotation = np.transpose(fold.X_rotation)
        Frame(fold.test_csv, fold.normalization.undo_from(fold.test_data.df))
        fold = Fold(self._repo, self._repo.K)
        Frame(self._repo.folder / 'undo_from.csv', fold.normalization.undo_from(fold.test_data.df))
        return self

    def _construct(self, folder: PathLike, X: NP.Matrix, function_vector: functions.Vector, noise: NP.Matrix, origin_meta: Dict[str, Any]) -> Repository:
        """ Construct Repository housing the sample design matrix ``(X, f(X) + noise)``.

        Args:
            folder: The Repository folder.
            X: An (N,M) design matrix of inputs.
            function_vector: An (L,) Vector of test.functions.
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

    def __init__(self, base_folder: PathLike, doe: DOE.Method, function_vector: functions.Vector, N: int, M: int, noise_variance: GaussianNoise.Variance,
                 overwrite_existing: bool = False, **kwargs: Any):
        """ Construct the folder of a test repo from information regarding the test sample and test functions.

        Args:
            base_folder: The base folder under which the Repository will sit.
            doe: An experimental design for the sample inputs.
            function_vector: A Vector of test.functions.
            N: The number of datapoints in the sample.
            M: The input dimensionality.
            noise_magnitude: The (L,L) homoskedastic GaussianNoise.Variance.
            overwrite_existing: Whether to overwrite an existing Repository.
            **kwargs: Options passed straight to doe.
        """
        self._N, self._noise_variance = N, noise_variance
        folder = Path(base_folder) / f'{function_vector.name}.M.{M:d}.{self._noise_variance}.N.{N:d}'
        if folder.is_dir() and not overwrite_existing:
            self._repo = Repository(folder)
        else:
            self._repo = self._construct(folder=folder, X=doe(N, M, **kwargs), function_vector=function_vector,
                                         noise=GaussianNoise(N, self._noise_variance())(repo=None),
                                         origin_meta={'DOE': doe.__name__, 'function_vector': function_vector.meta, 'noise': self._noise_variance.meta})
            Frame(folder / 'likelihood.variance.csv', pd.DataFrame(self._noise_variance()))


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Please provide a single argument, which is the base folder to which the latin hypercube csv will be written.')
    else:
        base_folder = Path(sys.argv[1])
        M = 7
        for N in (100, 150, 200):
            pd.DataFrame(DOE.latin_hypercube(N, M)).to_csv(base_folder / f'lhs{N}.csv')

