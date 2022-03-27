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


BASE_PATH = Path('C:\\Users\\fc1ram\\Documents\\Rom\\dat\\SoftwareTest\\7.5')


def fold_and_rotate_with_tests(repo: Repository, K: int, rotation: NP.Matrix):
    repo._data.df = repo._data.df * 5
    repo._data.write()
    fold_and_rotate(repo, K, rotation)
    shutil.copytree(repo.fold_folder(repo.K), repo.folder / f'fold.{repo.K + 1}')
    fold = Fold(repo, repo.K + 1)
    fold.X_rotation = np.transpose(rotation)
    Frame(fold._test_csv, fold.normalization.undo_from(fold._test_data.df))
    fold = Fold(repo, repo.K)
    Frame(repo.folder / 'undone.csv', fold.normalization.undo_from(fold.test_data.df))


def fold_and_rotate(repo: Repository, K: int, rotation: NP.Matrix):
    repo.into_K_folds(K)
    rng = range(repo.K + 1) if K > 1 else range(1,2)
    for k in rng:
        fold = Fold(repo, k)
        fold.X_rotation = rotation


# noinspection PyShadowingNames
def run_gpr(name, function_names: Sequence[str], N: int, noise_variance: [float], noise_label: str, random: bool, M: int = 5, K: int = 2):
    if isinstance(function_names, str):
        function_names = [function_names]
    f = tuple((functions.FunctionWithMeta.DEFAULT[function_name] for function_name in function_names))
    store_folder = '.'.join(function_names) + f'.{M:d}.{noise_label}.{N:d}'
    if random:
        rotation = scipy.stats.ortho_group.rvs(M)
        store_folder += '.rotated'
    else:
        rotation = np.eye(M)
    store_folder = BASE_PATH / store_folder
    repo = functions.sample(f, N, M, noise_variance, store_folder)
    # fold_and_rotate_with_tests(repo, K, rotation)
    fold_and_rotate(repo, K, rotation)
    run.gpr(name=name, repo=repo, is_read=None, is_isotropic=None, is_independent=True, kernel_parameters=None, parameters=None,
            optimize=True, test=True, analyze=False)


def run_gpr_gsa(name, function_names: Sequence[str], N: int, noise_variance: [float], noise_label: str, random: bool, M: int = 5, K: int = 2):
    if isinstance(function_names, str):
        function_names = [function_names]
    f = tuple((functions.FunctionWithMeta.DEFAULT[function_name] for function_name in function_names))
    store_folder = '.'.join(function_names) + f'.{M:d}.{noise_label}.{N:d}'
    if random:
        rotation = scipy.stats.ortho_group.rvs(M)
        store_folder += '.rotated'
    else:
        rotation = np.eye(M)
    store_folder = BASE_PATH / store_folder
    repo = functions.sample(f, N, M, noise_variance, store_folder)
    # fold_and_rotate_with_tests(repo, K, rotation)
    fold_and_rotate(repo, K, rotation)
    run.gpr(name=name, repo=repo, is_read=None, is_isotropic=None, is_independent=True, kernel_parameters=None, parameters=None,
            optimize=True, test=True, analyze=True)


# noinspection PyShadowingNames
def compare_gpr(name, function_names: Sequence[str], N: int, noise_variance: [float], noise_label: str, random: bool, M: int = 5):
    if isinstance(function_names, str):
        function_names = [function_names]
    f = tuple((functions.FunctionWithMeta.DEFAULT[function_name] for function_name in function_names))
    store_folder = '.'.join(function_names) + f'.{M:d}.{noise_label}.{N:d}'
    if random:
        store_folder += '.rotated'
    store_folder = BASE_PATH / store_folder
    repo = Repository(store_folder)
    run.gpr(name=name, repo=repo, is_read=None, is_isotropic=False, is_independent=True, kernel_parameters=None, parameters=None,
            optimize=True, test=True, analyze=False)


def run_gsa(name, function_names: Sequence[str], N: int, noise_variance: [float], noise_label: str, random: bool, M: int = 5):
    if isinstance(function_names, str):
        function_names = [function_names]
    f = tuple((functions.FunctionWithMeta.DEFAULT[function_name] for function_name in function_names))
    store_folder = '.'.join(function_names) + f'.{M:d}.{noise_label}.{N:d}'
    if random:
        store_folder += '.rotated'
    store_folder = BASE_PATH / store_folder
    repo = Repository(store_folder)
    run.gpr(name=name, repo=repo, is_read=None, is_isotropic=False, is_independent=True, kernel_parameters=None, parameters=None,
            optimize=False, test=False, analyze=True)


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


if __name__ == '__main__':
    data = sampling.latin_hypercube(1000, 3, is_centered=True)
    data = pd.DataFrame(data)
    data.to_csv(Path('C:\\Users\\fc1ram\\Documents\\Rom\\dat\\WellcomeLeap\\doe.csv'))
    # with run.Context('Test', float='float64', device='CPU'):  #
    #     for N in (1100, 1320, 1650): #
    #         for noise_magnitude in (0.001, 0.01, 0.05,): #  # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0
    #             noise_label = f'{noise_magnitude:.3f}'
    #             for random in (False, ):
    #                 for M in (5,):
    #                     # run_gpr('initial', ['ishigami'], N, noise_variance(L=1, scale=noise_magnitude, diagonal=True),
    #                     #         noise_label=noise_label, random=random, M=M, K=1)
    #                     with run.Timing(f'N={N}, noise={noise_magnitude}'):
    #                         run_gpr('initial', ['ishigami'], N, noise_variance(L=1, scale=noise_magnitude, diagonal=True),
    #                                 noise_label=noise_label, random=random, M=M, K=10)
    #                         run_gpr('initial', ['sobol_g'], N, noise_variance(L=1, scale=noise_magnitude, diagonal=True),
    #                                 noise_label=noise_label, random=random, M=M, K=10)
