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

""" Contains developer test utilities for romcomma. """

from romcomma.base.definitions import *
from romcomma import run
from romcomma.data.storage import Fold, Repository, Frame
from romcomma.test import functions
import shutil
import scipy.stats


def fold_and_rotate(repo: Repository, K: int, rotation: NP.Matrix, is_rotation_undone: bool=False) -> Repository:
    """ Fold and rotate a Repository. The rotation (like normalization) applies to each fold, not the repo itself.

    Args:
        repo: The Repository to fold.
        K: The number of Folds &ge 1.
        rotation: The (M,M) rotation matrix to apply to the inputs.
        is_rotation_undone: Whether to test the undo function in an extra Fold.
    """
    repo.into_K_folds(K)
    rng = range(repo.K + 1) if K > 1 else range(1,2)
    for k in rng:
        fold = Fold(repo, k)
        fold.X_rotation = rotation
    if is_rotation_undone:
        shutil.copytree(repo.fold_folder(repo.K), repo.folder / f'fold.{repo.K + 1}')
        fold = Fold(repo, repo.K + 1)
        fold.X_rotation = np.transpose(rotation)
        Frame(fold.test_csv, fold.normalization.undo_from(fold.test_data.df))
        fold = Fold(repo, repo.K)
        Frame(repo.folder / 'undo_from.csv', fold.normalization.undo_from(fold.test_data.df))
    return repo


def noise_variance(L: int, noise_magnitude: float, is_diagonal: bool = False, is_stochastic: bool = False) -> NP.Matrix:
    """ Generate a noise covariance matrix

    Args:
        L: Output dimension.
        noise_magnitude: The StdDev of noise.
        is_diagonal: True to return a diagonal variance matrix.
        is_stochastic: True to return a random matrix.
    Returns: An (L,L) covariance matrix.
    """
    if is_stochastic:
        result = 2 * np.random.random_sample((2, 2))
        result = np.matmul(result, result.transpose())
    else:
        result = -np.ones(shape=(L, L)) / 10 + 11 * np.eye(L) / 10
    if is_diagonal:
        result = np.diag(np.diag(result))
    return noise_magnitude * noise_magnitude * result


def noise_label(noise_magnitude: float, is_diagonal: bool = False, is_stochastic: bool = False) -> str:
    """ The label for noise, as used by repo_folder().

    Args:
        noise_magnitude: The StdDev of noise.
        is_diagonal: True to return a diagonal variance matrix.
        is_stochastic: True to return a random matrix.
    Returns: The label for noise.
    """
    prefix = ('d.' if is_diagonal else 'n.') + ('s.' if is_stochastic else 'n.')
    return f'{prefix}{noise_magnitude:.3f}'


def repo_folder(base_folder: PathLike, function_names: Union[str, Sequence[str]], N: int, M: int,
                noise_magnitude: float, is_noise_diagonal: bool = False, is_noise_variance_stochastic: bool = False, is_input_rotated: bool = False) -> Path:
    """ Construct the folder of a test repo from information regarding the test sample and test functions.

    Args:
        base_folder: The base folder under which the folder will sit.
        function_names: A sequence of test.functions names.
        N: The number of datapoints in the sample.
        M: The input dimensionality.
        noise_magnitude: The StdDev of noise.
        is_noise_diagonal: True for a diagonal noise variance matrix.
        is_noise_variance_stochastic: True for a random noise covariance matrix.
        is_input_rotated: True to randomly rotate the inputs.
    Returns: The folder of the test repo.
    """
    if isinstance(function_names, str):
        function_names = [function_names]
    folder = '.'.join(function_names) + f'.M.{M:d}.{noise_label(noise_magnitude, is_noise_diagonal, is_noise_variance_stochastic)}.N.{N:d}'
    return Path(base_folder) / (folder + '.r' if is_input_rotated else folder)


# noinspection PyShadowingNames
def sample(base_folder: PathLike, function_names: Union[str, Sequence[str]], N: int, M: int, K: int,
           noise_magnitude: float, is_noise_diagonal: bool = False, is_noise_variance_stochastic: bool = False, 
           is_input_rotated: bool = False, is_rotation_undone: bool = False) -> Repository:
    """
    
    Args:
        base_folder: The base folder under which the repo folder will sit.
        function_names: A sequence of test.functions names.
        N: The number of datapoints in the sample.
        M: The input dimensionality.
        K: The number of Folds in the repo.
        noise_magnitude: The StdDev of noise.
        is_noise_diagonal: True for a diagonal noise variance matrix.
        is_noise_variance_stochastic: True for a random noise covariance matrix.
        is_input_rotated: True to randomly rotate the inputs.
        is_rotation_undone: Whether to test the undo function in an extra Fold.
    Returns: The folder of the test repo.
    """
    if is_input_rotated:
        rotation = scipy.stats.ortho_group.rvs(M)
    else:
        rotation = np.eye(M)
    function_names = [function_names] if isinstance(function_names, str) else function_names
    functions_with_meta = tuple((functions.FunctionWithMeta.DEFAULT[function_name] for function_name in function_names))
    repo = fold_and_rotate(functions.sample(functions_with_meta, N, M,
                                            noise_variance(len(function_names), noise_magnitude, is_noise_diagonal, is_noise_variance_stochastic),
                                            repo_folder(base_folder, function_names, N, M, noise_magnitude, is_noise_diagonal, is_noise_variance_stochastic,
                                                        is_input_rotated)),
                           K, rotation, is_rotation_undone)
    return repo


def aggregator(base_folder: Union[Path, str], child_folder: Union[Path, str], function_names: Sequence[str],
               N: int, M: int, noise_magnitude: float, is_noise_diagonal: bool = False, is_noise_variance_stochastic: bool = False,
               is_input_rotated: bool = False) -> Dict[str, Any]:
    """ Construct an aggregator Dict for a repo_folder(...).

    Args:
        base_folder: The base folder containing the repo, under which the output will also sit.
        child_folder: The child folder within the repo where the .csv to aggregate sits. The aggregate will sit under  base_folder/child_folder
        function_names: A sequence of test.functions names.
        N: The number of datapoints in the sample.
        M: The input dimensionality.
        noise_magnitude: The StdDev of noise.
        is_noise_diagonal: True for a diagonal noise variance matrix.
        is_noise_variance_stochastic: True for a random noise covariance matrix.
        is_input_rotated: True to randomly rotate the inputs.
    Returns: The aggregator for the repo_folder specified.
    """
    return {'folder': repo_folder(base_folder, function_names, N, M, noise_magnitude, is_noise_diagonal, is_noise_variance_stochastic, is_input_rotated)
                      / child_folder, 'N': N, 'noise': noise_magnitude}
