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

""" Contains developer tests of romcomma. """
import pandas as pd

from romcomma.base.definitions import *
from romcomma import run
from romcomma.data.storage import Fold, Repository, Frame
from romcomma.test import functions, sampling
import shutil
import scipy.stats


BASE_PATH = Path('C:\\Users\\fc1ram\\Documents\\Rom\\dat\\SoftwareTest\\9s\\9.5')


def fold_and_rotate_with_tests(repo: Repository, K: int, rotation: NP.Matrix):
    repo._data.df = repo._data.df * 5
    repo._data.write()
    fold_and_rotate(repo, K, rotation)
    shutil.copytree(repo.fold_folder(repo.K), repo.folder / f'fold.{repo.K + 1}')
    fold = Fold(repo, repo.K + 1)
    fold.X_rotation = np.transpose(rotation)
    Frame(fold.test_csv, fold.normalization.undo_from(fold._test_data.df))
    fold = Fold(repo, repo.K)
    Frame(repo.folder / 'undone.csv', fold.normalization.undo_from(fold.test_data.df))


def fold_and_rotate(repo: Repository, K: int, rotation: NP.Matrix):
    repo.into_K_folds(K)
    rng = range(repo.K + 1) if K > 1 else range(1,2)
    for k in rng:
        fold = Fold(repo, k)
        fold.X_rotation = rotation


# noinspection PyShadowingNames
def run_gpr(name, function_names: Sequence[str], N: int, noise_variance: [float], noise_label: str, random: bool, M: int = 5, K: int = 1):
    if isinstance(function_names, str):
        function_names = [function_names]
    f = tuple((functions.FunctionWithMeta.DEFAULT[function_name] for function_name in function_names))
    if random:
        rotation = scipy.stats.ortho_group.rvs(M)
    else:
        rotation = np.eye(M)
    repo = functions.sample(f, N, M, noise_variance, repo_folder(function_names, N, noise_label, random, M))
    # fold_and_rotate_with_tests(repo, K, rotation)
    fold_and_rotate(repo, K, rotation)
    run.gpr(name=name, repo=repo, is_read=None, is_isotropic=None, is_independent=True, kernel_parameters=None, parameters=None,
            optimize=True, test=True)


# noinspection PyShadowingNames
def compare_gpr(name, function_names: Sequence[str], N: int, noise_label: str, random: bool, M: int = 5):
    if isinstance(function_names, str):
        function_names = [function_names]
    f = tuple((functions.FunctionWithMeta.DEFAULT[function_name] for function_name in function_names))
    repo = Repository(repo_folder(function_names, N, noise_label, random, M))
    run.gpr(name=name, repo=repo, is_read=None, is_isotropic=False, is_independent=None, kernel_parameters=None, parameters=None,
            optimize=False, test=False)


def run_gsa(name, function_names: Sequence[str], N: int, noise_label: str, random: bool, M: int = 5, is_independent: bool = True,
            ignore_exceptions: bool = False, **kwargs: Any):
    if isinstance(function_names, str):
        function_names = [function_names]
    f = tuple((functions.FunctionWithMeta.DEFAULT[function_name] for function_name in function_names))
    repo = Repository(repo_folder(function_names, N, noise_label, random, M))
    run.gsa(name=name, repo=repo, is_independent=is_independent, ignore_exceptions=ignore_exceptions, **kwargs)


def noise_variance(L: int, scale: float, diagonal: bool = False, random: bool = False):
    if diagonal:
        result = np.eye(L)
    elif random:
        result = np.random.random_sample((2, 2))
        result = np.matmul(result, result.transpose())
        print(scale * scale * result)
    else:
        result = np.ones(shape=(L, L))/2 + np.eye(L)/2
    return scale * scale * result


def repo_folder(function_names: Sequence[str], N: int, noise_label: str, random: bool, M: int = 5) -> Path:
    if isinstance(function_names, str):
        function_names = [function_names]
    folder = '.'.join(function_names) + f'.{M:d}.{noise_label}.{N:d}'
    if random:
        folder += '.rotated'
    return BASE_PATH / folder


def noise_label(noise_magnitude: float) -> str:
    return f'{noise_magnitude:.3f}'


def aggregator(child_path: Union[Path, str], function_names: Sequence[str], N: int, noise_magnitude: float, random: bool, M: int = 5) -> Dict[str, Any]:
    return {'path': repo_folder(function_names, N, noise_label(noise_magnitude), random, M) / child_path, 'N': N, 'noise': noise_magnitude}


def aggregate(aggregators: Dict[str, Sequence[Dict[str, Any]]], dst: Union[Path, str], **kwargs):
    """ Aggregate csv files over aggregators.

    Args:
        aggregators: A Dict of aggregators, keyed by csv filename. An aggregator is a List of Dicts containing source path ['path']
            and {key: value} to insert column 'key' and populate it with 'value' in path/csv.
        dst: The destination path, to which csv files listed as the keys in aggregators.
        **kwargs: Write options passed directly to pd.Dataframe.to_csv(). Overridable defaults are {'index': False, 'float_format':'%.6f'}
    """
    dst = Path(dst)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(mode=0o777, parents=True, exist_ok=False)
    for csv, aggregator in aggregators.items():
        is_initial = True
        results = None
        for file in aggregator:
            result = pd.read_csv(Path(file.pop('path'))/csv)
            for key, value in file.items():
                result.insert(0, key, np.full(result.shape[0], value), True)
            if is_initial:
                results = result.copy(deep=True)
                is_initial = False
            else:
                results = pd.concat([results, result.copy(deep=True)], axis=0, ignore_index=True)
            kwargs = {'index': False, 'float_format': '%.6f'} | kwargs
        results.to_csv(dst/csv, index=False)


if __name__ == '__main__':
    with run.Context('Test', float='float64', device='CPU'):  #
        for N in (40, 60, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000,):   #
            for noise_magnitude in (0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0):    #
                for random in (False, ):
                    for is_T_partial in (True, False):
                        with run.Timing(f'N={N}, noise={noise_magnitude}, is_T_partial={is_T_partial}'):
                            run_gsa('initial', ['ishigami', 'sobol_g', 'sobol_g2'], N, noise_label(noise_magnitude), random, 5,
                                    is_independent=True, ignore_exceptions=True, is_T_partial=is_T_partial)
