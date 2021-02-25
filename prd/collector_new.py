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
    concatenate, savetxt, loadtxt
from numpy.linalg import norm, eigvalsh
from pathlib import Path
from pandas import concat, DataFrame, read_csv
from json import load
import shutil
from time import time

EFFECTIVELY_ZERO = 1.0E-64
BASE_PATH = Path('X:\\comma_group1\\Rom\\dat\\TestFunctions\\ROM\\0')
RESULTS_PATH = BASE_PATH / "results"
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
        return 3
    elif test_function == "ishigami":
        return 3
    elif test_function == "sin.2":
        return 2
    else:
        return 1


# def _run_test(test_function: str, N: int, noise_std: float, random: bool, gps: Tuple[str, ...], M: int):
#     store = store_path(test_function, N, noise_std, random, M)
#     Mu = choose_Mu(test_function)
#     kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, Mu), 0.2, dtype=float))
#     parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-5, e=1E-10)
#     for k in range(K):
#         fold = data.Fold(store, k, Mu)
#         for gp in gps:
#             dst = fold.dir / "{0}.reduced".format(gp)
#             if dst.exists():
#                 shutil.rmtree(dst)
#             shutil.copytree(src=fold.dir / gp, dst=dst)
#             gp = model.gpy_.GP(fold, dst.name, parameters)
#             gp.optimize()
#             gp.test()
#             model.gpy_.Sobol(gp)
#             gp = None


def _run_test(test_function: str, N: int, noise_std: float, random: bool, M: int):
    store = data.Store(store_path(test_function, N, noise_std, random, M))
    Mu = choose_Mu(test_function)
    gp_optimizer_options = {'optimizer': 'bfgs', 'max_iters': 5000, 'gtol': 1E-16}
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, 1), 2.5**(M/5), dtype=float))
    parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-6, e=0.003)
    name = 'rom.reduced'
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=Mu, parameters=parameters, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=Mu, parameters=None, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options, make_ard=True)


def run_tests(test_functions: Tuple[str, ...], Ns: Tuple[int, ...], noise_stds: Tuple[float, ...], randoms: Tuple[bool, ...],
              gps: Tuple[str, ...], Ms: Tuple[int, ...] = (5, )):
    for M in Ms:
        for N in Ns:
            for test_function in test_functions:
                for noise_std in noise_stds:
                    for random in randoms:
                        _run_test(test_function, N, noise_std, random, M)


def _test_stats(k: int, gp_path: Path) -> data.Frame:
    test = data.Frame(gp_path / "__test__.csv").df.copy()
    Y = test['Y'].values
    mean_ = test['Predictive Mean'].values
    std = test['Predictive Std'].values
    err = abs(Y - mean_)
    outliers = floor_divide(err, 2 * std)
    df = DataFrame({'fold': k, 'RMSE': sqrt(mean(err ** 2)) / 4, 'Prediction Std': mean(std),
                         'Outliers': count_nonzero(outliers) / len(std)}, index=[0])
    return data.Frame(gp_path / "test_stats.csv", df)


def _collect_test_stats(test_function: str, N: int, noise_std: float, random: bool, gps: Tuple[str, ...], M: int):
    store = store_path(test_function, N, noise_std, random, M)
    for k in range(K):
        fold = data.Fold(store, k)
        for gp in gps:
            gp_path = fold.dir / gp
            frame = _test_stats(k, gp_path)


def collect_tests(test_functions: Tuple[str, ...], Ns: Tuple[int, ...], noise_stds: Tuple[float, ...], randoms: Tuple[bool, ...],
                  gps: Tuple[str, ...], Ms: Tuple[int, ...] = (5, )):
    for M in Ms:
        for N in Ns:
            for test_function in test_functions:
                for noise_std in noise_stds:
                    for random in randoms:
                            _collect_test_stats(test_function, N, noise_std, random, gps, M)


def _collect_std(test_function: str, N: int, noise_std: float, random: bool, M: int):
    store = data.Store(store_path(test_function, N, noise_std, random, M))
    destination = store.dir / "results"
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(mode=0o777, parents=True, exist_ok=False)
    result = 0.0
    for k in range(K):
        fold = data.Fold(store, k)
        result += fold.standard.df.iloc[-1, -1]/K
    savetxt(fname=(destination / "std.csv"), X=atleast_2d(result), delimiter=",")


def _collect_result(test_function: str, N: int, noise_std: float, random: bool, gps: Tuple[str, ...], M: int):
    store = store_path(test_function, N, noise_std, random, M)
    destination = store / "results"
    destination.mkdir(mode=0o777, parents=True, exist_ok=True)
    for gp in gps:
        for sobol in (True, False):
            if sobol:
                lin_trans = linear_transformation(store)
                frame = data.Frame(destination / "{0}.{1}".format(gp, "True_Theta.csv"),  DataFrame(lin_trans))
                lin_trans = transpose(lin_trans)
                params = ("Theta.csv", "S.csv", "S1.csv")
            else:
                params = ("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv", "test_stats.csv")
            for param in params:
                results = None
                avg = None
                for k in range(K):
                    source = (store / "fold.{0:d}".format(k)) / gp
                    source = source / "sobol" if sobol else source / "kernel" if param == "lengthscale.csv" else source
                    result = data.Frame(source / param, **model.base.Model.CSV_PARAMETERS).df.copy(deep=True)
                    result.insert(0, "fold", full(result.shape[0], k), True)
                    if k == 0:
                        results = result
                        avg = result / K
                    else:
                        results = concat([results, result], axis=0, ignore_index=True, sort=False)
                        avg += result / K
                avg.loc[:, 'fold'] = 'mean'
                results = concat([results, avg], axis=0, ignore_index=True, sort=False)
                frame = data.Frame(destination / "{0}.{1}".format(gp, param), results)


def _singular_values(matrix: NP.Matrix) -> NP.Covector:
    matrix = matrix @ transpose(matrix)
    return atleast_2d(sqrt(eigvalsh(matrix)))


def _analyze_theta(test_function: str, N: int, noise_std: float, M: int) -> DataFrame:
    Mu = choose_Mu(test_function)
    random = {flag: (store_path(test_function, N, noise_std, flag, M) / "results") for flag in (True, False)}
    theta_df_random = {flag: data.Frame(random[flag] / "rom.optimized.Theta.csv", **model.base.Model.CSV_PARAMETERS).df for flag in (True, False)}
    theta_true = data.Frame(random[True] / "rom.optimized.True_Theta.csv", **model.base.Model.CSV_PARAMETERS).df.values
    for k in range(K):
        theta_random = {flag: theta_df_random[flag].loc[theta_df_random[flag]['fold'] == str(k)].values[:, 1:].copy().astype(float)
                        for flag in (True, False)}
        h = theta_random[False] @ theta_true @ theta_random[True].transpose()
        resultA = _singular_values(h[:Mu, :Mu])
        resultI = _singular_values(h[Mu:, Mu:])
        result = concatenate((resultA, resultI), axis=1)
        result_df = DataFrame(result)
        result_df.insert(0, "fold", full(result.shape[0], k, dtype=int), True)
        if k == 0:
            results = DataFrame(result_df)
            mean = results.copy(deep=True) / K
        else:
            results = concat([results, result_df], axis=0, ignore_index=True, sort=False)
            mean += result_df / K
    mean.loc[:, 'fold'] = 'mean'
    results = concat([results, mean], axis=0, ignore_index=True, sort=False)
    results.to_csv(random[True] / "rom.optimized.reduced.Theta_Analyzed.csv")
    return mean


def collect_results(test_functions: Tuple[str, ...], Ns: Tuple[int, ...], noise_stds: Tuple[float, ...], randoms: Tuple[bool, ...],
                  gps: Tuple[str, ...], Ms: Tuple[int, ...] = (5, )):
    for M in Ms:
        for N in Ns:
            for test_function in test_functions:
                for noise_std in noise_stds:
                    for random in randoms:
                        _collect_std(test_function, N, noise_std, random, M)
                        _collect_test_stats(test_function, N, noise_std, random, gps, M)
                        _collect_result(test_function, N, noise_std, random, gps, M)
                    _analyze_theta(test_function, N, noise_std, M)


def summarise_results(test_functions: Tuple[str, ...], Ns: Tuple[int, ...], noise_stds: Tuple[float, ...], randoms: Tuple[bool, ...],
                      gps: Tuple[str, ...], Ms: Tuple[int, ...] = (5, )):
    for M in Ms:
        for test_function in test_functions:
            Mu = choose_Mu(test_function)
            for random in randoms:
                destination = RESULTS_PATH / "{0}.{1:d}.{2}".format(test_function, M, _random_str(random))
                destination.mkdir(mode=0o777, parents=True, exist_ok=True)
                for gp in gps:
                    for sobol in (True, False):
                        if sobol:
                            params = (("S.csv", "S1.csv", "Theta.csv", "True_Theta.csv", "Theta_Analyzed.csv")
                                      if random and gp == "rom.optimized.reduced" else ("S.csv", "S1.csv", "Theta.csv"))
                        else:
                            params = ("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv", "test_stats.csv")
                        for param in params:
                            is_initial = True
                            for N in Ns:
                                for noise_std in noise_stds:
                                    results_path = store_path(test_function, N, noise_std, random, M) / "results"
                                    std = loadtxt(results_path / 'std.csv')
                                    source = results_path / "{0}.{1}".format(gp, param)
                                    result = data.Frame(source, **model.base.Model.CSV_PARAMETERS).df.copy(deep=True)
                                    result = (result.copy(deep=True) if param == "True_Theta.csv"
                                              else result.loc[result['fold'] == 'mean'].drop('fold', axis=1).copy(deep=True))
                                    result.insert(0, "N", full(result.shape[0], int(N/2), dtype=int), True)
                                    result.insert(0, "Noise", full(result.shape[0], noise_std/std), True)
                                    if is_initial:
                                        results = result
                                        is_initial = False
                                    else:
                                        results = concat([results, result], axis=0, ignore_index=True, sort=False)
                            results.to_csv(destination / "{0}.{1}".format(gp, param), index=False)
                            results.to_csv(destination / "formatted.{0}.{1}".format(gp, param), float_format='%.4f', index=False)


def amalgamate_tables(test_functions: Tuple[str, ...], params: Tuple[str, ...], randoms: Tuple[bool, ...],
                      gps: Tuple[str, ...], Ms: Tuple[int, ...] = (5,) ):
    for M in Ms:
        for test_function in test_functions:
            random = {flag: RESULTS_PATH / "{0}.{1:d}.{2}".format(test_function, M, _random_str(flag)) for flag in randoms}
            for flag in randoms:
                for param in params:
                    is_initial = True
                    for gp in gps:
                        result = read_csv(random[flag] / "{0}.{1}".format(gp, param), index_col=False)
                        if param == "test_stats.csv":
                            result = result.drop(columns=['fold.1'])
                        result = result.values * 100
                        if is_initial:
                            results = result.copy()
                            results[:, 1] /= 100
                            is_initial = False
                        else:
                            results = concatenate((results, result[:, 2:]), axis=1)
                        fmt = ("%4.2f", "%d") + ("%4.2f",) * (results.shape[1]-2)
                    savetxt(fname=random[flag] / "{0}.{1}".format("amalgamated", param), X=results, fmt=fmt, delimiter=",")


def amalgamate_thetas(test_functions: Tuple[str, ...], Ms: Tuple[int, ...] = (5,) ):
    for M in Ms:
        for test_function in test_functions:
            random = {flag: RESULTS_PATH / "{0}.{1:d}.{2}".format(test_function, M, _random_str(flag)) for flag in (True, False)}
            result = {flag: read_csv(random[flag] / "amalgamated.S.csv", index_col=False, header=None).values for flag in (True, False)}
            results = concatenate((result[False], result[True][:, 2:]), axis=1)
            fmt = ("%4.2f", "%d") + ("%4.2f",) * (results.shape[1]-2)
            savetxt(fname=(random[True] / "amalgamated.amalgamated.S.csv"), X=results, fmt=fmt, delimiter=",")


def copy_results():
    destination = DOC_PATH / "results"
    shutil.rmtree(destination, ignore_errors=False)
    shutil.copytree(RESULTS_PATH, destination)


if __name__ == '__main__':
    start_time = time()
    intermediate_time = time()
    run_tests(test_functions=("sobol_g", ), Ns=(200, 400, 800, 1600, 3200), noise_stds=(0.1, 0.05, 0.01),
              randoms=(False, True), gps=('rbf.ard', 'rom.optimized'), Ms=(5,))
    print("tests finished in {0:.1f} minutes.".format((time() - intermediate_time) / 60))
    intermediate_time = time()
    collect_results(test_functions=("sin.1", "sin.2", "ishigami", "sobol_g"), Ns=(200, 400, 800, 1600, 3200), noise_stds=(0.1, 0.05, 0.01),
                    randoms=(False, True), gps=('rbf.ard', 'rom.optimized', 'rom.reduced', 'rom.optimized.reduced'), Ms=(5,))
    print("collecting results finished in {0:.1f} minutes.".format((time() - intermediate_time) / 60))
    intermediate_time = time()
    summarise_results(test_functions=("sin.1", "sin.2", "ishigami", "sobol_g"), Ns=(200, 400, 800, 1600, 3200), noise_stds=(0.1, 0.05, 0.01),
                      randoms=(False, True), gps=('rbf.ard', 'rom.optimized', 'rom.reduced', 'rom.optimized.reduced'), Ms=(5,))
    print("summarising results finished in {0:.1f} minutes.".format((time() - intermediate_time) / 60))
    amalgamate_tables(test_functions=("sin.1", "sin.2", "ishigami", "sobol_g"), params=("Theta_Analyzed.csv",), randoms=(True,),
                      gps=('rom.optimized.reduced',), Ms=(5,))
    amalgamate_tables(test_functions=("sin.1", "sin.2", "ishigami", "sobol_g"), params=("test_stats.csv", "S.csv"), randoms=(True, False),
                      gps=('rbf.ard', 'rom.optimized', 'rom.reduced', 'rom.optimized.reduced'), Ms=(5,))
    amalgamate_thetas(test_functions=("sin.1", "sin.2", "ishigami", "sobol_g"), Ms=(5,))
    copy_results()

