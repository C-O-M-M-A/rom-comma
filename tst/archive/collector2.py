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

""" Run this module first thing, to test your installation of romcomma.

**Contents**:
    **predict**: Prediction using a GaussianBundle.

    **test_input**: A rudimentary test input, for installation testing.
"""

from romcomma import distribution, function, data, model
from romcomma.typing_ import NP, Tuple
from numpy import zeros, eye, pi, full, array, transpose, diag, sign, ones, atleast_2d, abs, floor_divide, count_nonzero, mean, sqrt, \
    concatenate, savetxt
from numpy.linalg import norm, eigvalsh
from pathlib import Path
from pandas import concat, DataFrame, read_csv
from json import load
import shutil
from time import time

EFFECTIVELY_ZERO = 1.0E-64
BASE_PATH = Path('X:\\comma_group1\\Rom\\dat\\TestFunctions\\GP\\1')
RESULTS_PATH = BASE_PATH / "results"
SUMMARY_PATH = BASE_PATH / "summary"
DOC_PATH = Path('X:\\comma_group1\\Rom\\doc\\Papers\\romgp-paper-1')
K = 2

def linear_transformation(model_dir: Path) -> NP.Matrix:
    with open(model_dir / "__meta__.json", mode='r') as file:
        meta = load(file)
    function_with_parameters = meta['origin']['functions_with_parameters'][0].split("; matrix=")
    if len(function_with_parameters) > 1:
        function_with_parameters = eval(function_with_parameters[-1][:-1])
        return array(function_with_parameters)
    else:
        return ones((meta['data']['M'], meta['data']['M']), dtype=float)


def _random_str(random: bool) -> str:
    return "random" if random else "rom"


def store_path(test_function: str, N: int, noise_std: float, random: bool, M: int = 5) -> Path:
    return BASE_PATH / (test_function + '.{0:d}.{1:.3f}.{2:d}.'.format(M, noise_std, N) + _random_str(random))


def choose_Mu(test_function: str) -> int:
    if test_function == "sobol_g":
        return 4
    elif test_function == "ishigami":
        return 3
    elif test_function == "sin.2":
        return 2
    else:
        return 1


def _test_stats(k: int, test: DataFrame) -> DataFrame:
    Y = test['Y'].values
    mean_ = test['Predictive Mean'].values
    std = test['Predictive Std'].values
    err = abs(Y - mean_)
    outliers = floor_divide(err, 2 * std)
    return DataFrame({'fold': k, 'RMSE': sqrt(mean(err ** 2)), 'Prediction Std': sqrt(mean(std ** 2)),
                         'Outliers': count_nonzero(outliers) / len(std)}, index=[0])


def _collect_test_stats(test_function: str, N: int, noise_std: float, random: bool, gp: str, M: int = 5):
    source_store = store_path(test_function, N, noise_std, random, M)
    for k in range(K):
        fold = data.Fold(source_store, k)
        gp_dir = fold.dir / gp
        frame = data.Frame(gp_dir / "test_stats.csv", _test_stats(k, data.Frame(gp_dir / "__test__.csv").df.copy()))


def collect_tests(test_functions: Tuple[str, ...], Ns: Tuple[int, ...], noise_stds: Tuple[float, ...], randoms: Tuple[bool, ...], 
                  gps: Tuple[str, ...], Ms: Tuple[int, ...] = (5, )):
    for M in Ms:
        for N in Ns:
            for test_function in test_functions:
                for noise_std in noise_stds:
                    for random in randoms:
                        for gp in gps:
                            _collect_test_stats(test_function, N, noise_std, random, gp, M)


def _collect_result(test_function: str, N: int, noise_std: float, random: bool, gp: str, M: int = 5):
    source_store = store_path(test_function, N, noise_std, random, M)
    destination = RESULTS_PATH / source_store.name
    destination.mkdir(mode=0o777, parents=True, exist_ok=True)
    for sobol in (False, ):
        if sobol:
            lin_trans = linear_transformation(source_store)
            frame = data.Frame(destination / "{0}.{1}".format(gp, "True_Theta.csv"),  DataFrame(lin_trans))
            lin_trans = transpose(lin_trans)
            params = ("Theta.csv", "S.csv", "S1.csv")
        else:
            params = (("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv") if gp == "rom.optimized"
                      else ("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv", "test_stats.csv"))
        for param in params:
            results = None
            avg = None
            for k in range(K):
                source = (source_store / "fold.{0:d}".format(k)) / gp
                source = source / "sobol" if sobol else source / "kernel" if param == "lengthscale.csv" else source
                result = data.Frame(source / param, **model.base.Model.CSV_PARAMETERS).df.copy(deep=True)
                # if param == "Theta.csv": # TODO: May not need this
                #     signs = result.values @ lin_trans
                #     signs = sign(diag(signs))
                #     signs.shape = (signs.shape[0], 1)
                #     result *= signs
                result.insert(0, "fold", full(result.shape[0], k), True)
                if k == 0:
                    results = result
                    avg = result / K
                else:
                    results = concat([results, result], axis=0, ignore_index=True)
                    avg += result / K
            avg.loc[:, 'fold'] = 'mean'
            results = concat([results, avg], axis=0, ignore_index=True)
            frame = data.Frame(destination / "{0}.{1}".format(gp, param), results)


def collect_results(test_functions: Tuple[str, ...], Ns: Tuple[int, ...], noise_stds: Tuple[float, ...], randoms: Tuple[bool, ...], 
                    gps: Tuple[str, ...], Ms: Tuple[int, ...] = (5,)):
    for M in Ms:
        for N in Ns:
            for test_function in test_functions:
                for noise_std in noise_stds:
                    for random in randoms:
                        for gp in gps:
                            _collect_result(test_function, N, noise_std, random, gp, M)


def summarise_results(test_functions: Tuple[str, ...], Ns: Tuple[int, ...], noise_stds: Tuple[float, ...], randoms: Tuple[bool, ...], 
                      gps: Tuple[str, ...], Ms: Tuple[int, ...] = (5,)):
    for test_function in test_functions:
            for random in randoms:
                destination = SUMMARY_PATH / (test_function + '.' + _random_str(random))
                destination.mkdir(mode=0o777, parents=True, exist_ok=True)
                for gp in gps:
                    for sobol in (False, ):
                        if sobol:
                            params = () if gp == "rom.reduced" else ("S.csv", "S1.csv", "Theta.csv", "True_Theta.csv", "Theta_Analyzed.csv")
                        else:
                            params = (("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv") if gp == "rom.optimized"
                                      else ("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv", "test_stats.csv"))
                        for param in params:
                            if (param == "Theta_Analyzed.csv" and ((not random) or (gp == "ard"))):
                                continue
                            is_initial = True
                            for M in Ms:
                                Mu = M
                                for N in Ns:
                                    for noise_std in noise_stds:
                                        source_store = store_path(test_function, N, noise_std, random, M)
                                        source = (RESULTS_PATH / source_store.name) / "{0}.{1}".format(gp, param)
                                        result = data.Frame(source, **model.base.Model.CSV_PARAMETERS).df.copy(deep=True)
                                        result = (result.copy(deep=True) if param == "True_Theta.csv"
                                                  else result.loc[result['fold'] == 'mean'].drop('fold', axis=1).copy(deep=True))
                                        result.insert(0, "N", full(result.shape[0], int(N/2), dtype=int), True)
                                        result.insert(0, "Noise", full(result.shape[0], noise_std), True)
                                        result.insert(0, "M", full(result.shape[0], M), True)
                                        if is_initial:
                                            results = result
                                            is_initial = False
                                        else:
                                            results = concat([results, result], axis=0, ignore_index=True, sort=False)
                            results.to_csv(destination / "{0}.{1}".format(gp, param), index=False)


def synopsise(test_functions: Tuple[str, ...], randoms: Tuple[bool, ...], gps: Tuple[str, ...]):
    destination = SUMMARY_PATH / "synopsis"
    destination.mkdir(mode=0o777, parents=True, exist_ok=True)
    params = ("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv", "test_stats.csv")
    for param in params:
        is_initial = True
        for test_function in test_functions:
            for random in randoms:
                source_path = SUMMARY_PATH / (test_function + '.' + _random_str(random))
                for gp in gps:
                    result = data.Frame(source_path / "{0}.{1}".format(gp, param), header=0, index_col=False).df.copy(deep=True)
                    result.insert(0, "GP", full(result.shape[0], gp), True)
                    result.insert(0, "Random Rotation", full(result.shape[0], random), True)
                    result.insert(0, "Test Function", full(result.shape[0], test_function), True)
                    if is_initial:
                        results = result
                        is_initial = False
                    else:
                        results = concat([results, result], axis=0, ignore_index=True, sort=False)
        results.to_csv(destination / "{0}".format(param), index=False)


if __name__ == '__main__':
    start_time = time()
    intermediate_time = time()
    collect_tests(test_functions=("sin.1", "sin.2", "ishigami", "sobol_g"), Ns=(200, 400, 800, 1600, 3200),
                  noise_stds=(0.000, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1), randoms=(True, False), gps=('rbf', 'rbf.ard'), Ms=(5, 10, 15))
    print("collecting tests finished in {0:.1f} minutes.".format((time() - intermediate_time) / 60))
    intermediate_time = time()
    collect_results(test_functions=("sin.1", "sin.2", "ishigami", "sobol_g"), Ns=(200, 400, 800, 1600, 3200),
                    noise_stds=(0.000, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1), randoms=(True, False), gps=('rbf', 'rbf.ard'), Ms=(5, 10, 15))
    print("collecting results finished in {0:.1f} minutes.".format((time() - intermediate_time) / 60))
    intermediate_time = time()
    summarise_results(test_functions=("sin.1", "sin.2", "ishigami", "sobol_g"), Ns=(200, 400, 800, 1600, 3200),
                      noise_stds=(0.000, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1), randoms=(True, False), gps=('rbf', 'rbf.ard'), Ms=(5, 10, 15))
    print("summarising results finished in {0:.1f} minutes.".format((time() - intermediate_time) / 60))
    print("main finished in {0:.1f} minutes.".format((time() - start_time) / 60))
    synopsise(test_functions=("sin.1", "sin.2", "ishigami", "sobol_g"), randoms=(True, False), gps=('rbf', 'rbf.ard'))
