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
BASE_PATH = Path('X:\\comma_group1\\Rom\\dat\\TestFunctions\\Scalar.2')
NOISELESS_DIR = 'Noiseless'
NORMAL_DIR = 'Normal'
UNIFORM_DIR = 'Uniform'
NORMAL_CDF_DIR = 'NormalCDF'
FOLDS = 2
DOC_PATH = Path('X:\\comma_group1\\Rom\\doc\\Papers\\romgp-paper-1')


def store_dir(store_name: str, noise_std: float, CDF_scale: NP.Array=None) -> Path:
    if noise_std <= EFFECTIVELY_ZERO:
        return BASE_PATH / NOISELESS_DIR / store_name
    elif CDF_scale is None:
        return BASE_PATH / NORMAL_DIR / store_name
    else:
        return BASE_PATH / NORMAL_CDF_DIR / store_name


def scalar_function_of_normal(store_name: str, N: int, M: int, X_std: float, noise_std: float, CDF_scale: NP.Array=None, CDF_loc: NP.Array=None,
                        pre_function_with_parameters: function.CallableWithParameters = None,
                       function_with_parameters: function.CallableWithParameters = None) -> data.Store:
    X_marginal = distribution.Univariate('norm', loc=0, scale=X_std)
    X_dist = distribution.Multivariate.Independent(M=M, marginals=X_marginal)
    noise_dist = (distribution.Multivariate.Normal(mean=zeros(1, dtype=float), covariance=noise_std ** 2 * eye(1, dtype=float))
                  if noise_std > EFFECTIVELY_ZERO else None)
    return function.sample(store_dir=store_dir(store_name, noise_std, CDF_scale), N=N, X_distribution=X_dist,
                               X_sample_design=distribution.SampleDesign.LATIN_HYPERCUBE, CDF_scale=CDF_scale,
                               CDF_loc=CDF_loc, pre_function_with_parameters=pre_function_with_parameters,
                               functions_with_parameters=function_with_parameters,
                               noise_distribution=noise_dist, noise_sample_design=distribution.SampleDesign.LATIN_HYPERCUBE)


def reverse_matrix(M: int) -> NP.Matrix:
    result = zeros((M, M), dtype=float)
    for i in range(M):
        result[i, M-i-1] = 1.0
    return result


def linear_transformation(model_dir: Path) -> NP.Matrix:
    with open(model_dir / "__meta__.json", mode='r') as file:
        meta = load(file)
    function_with_parameters = meta['origin']['functions_with_parameters'][0].split("; matrix=")
    if len(function_with_parameters) > 1:
        function_with_parameters = eval(function_with_parameters[-1][:-1])
        return array(function_with_parameters)
    else:
        return ones((meta['data']['M'], meta['data']['M']), dtype=float)


def store_dir(M: int, N: int, function_name: str, random: bool, noisy: bool) -> Path:
    name = function_name + '.{0:d}.{1:d}'.format(N, M) + ('.random' if random else '.rom')
    return (BASE_PATH / (NORMAL_CDF_DIR if noisy else NOISELESS_DIR)) / name


def choose_Mu(function_name: str) -> int:
    if function_name == "sobol_g":
        return 4
    elif function_name == "ishigami":
        return 3
    elif function_name == "sin.2":
        return 2
    else:
        return 1


def _test_stats(k: int, test: DataFrame) -> DataFrame:
    Y = test['Y'].values
    mean_ = test['Predictive Mean'].values
    std = test['Predictive Std'].values
    err = abs(Y - mean_)
    outliers = floor_divide(err, 2 * std)
    return DataFrame({'fold': k, 'RMSE': sqrt(mean(err ** 2)) / 4, 'Prediction Std': mean(std),
                         'Outliers': count_nonzero(outliers) / len(std)}, index=[0])


def _collect_test_stats(M, N, function_name, random, noisy):
    noisy_str = NORMAL_CDF_DIR if noisy else NOISELESS_DIR
    source_store = store_dir(M, N, function_name, random, noisy)
    for k in range(FOLDS):
        fold = data.Fold(source_store, k)
        gp_dir = fold.dir / "ard"
        frame = data.Frame(gp_dir / "test_stats.csv", _test_stats(k, data.Frame(gp_dir / "__test__.csv").df.copy()))


def _run_test(M, N, function_name, random, noisy):
    noisy_str = NORMAL_CDF_DIR if noisy else NOISELESS_DIR
    source_store = store_dir(M, N, function_name, random, noisy)
    Mu = choose_Mu(function_name)
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, Mu), 0.2, dtype=float))
    parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-5, e=1E-10)
    for k in range(FOLDS):
        fold = data.Fold(source_store, k, Mu)
        dst = fold.dir / "rom.reduced"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src=fold.dir / "rom.optimized", dst=dst)
        gp = model.gpy_.GP(fold, "rom.reduced", parameters)
        gp.optimize(model.gpy_.GP.DEFAULT_OPTIMIZER_OPTIONS)
        frame = data.Frame(gp.dir / "test_stats.csv", _test_stats(k, gp.test().df.copy()))


def run_tests(Ms: Tuple[int], Ns: Tuple[int], function_names: Tuple[str]):
    for M in Ms:
        for N in Ns:
            for function_name in function_names:
                for noisy in (True, False):
                    for random in (True, False):
                        _run_test(M, N, function_name, random, noisy)


def collect_tests(Ms: Tuple[int], Ns: Tuple[int], function_names: Tuple[str]):
    for M in Ms:
        for N in Ns:
            for function_name in function_names:
                for noisy in (True, False):
                    for random in (True, False):
                        _collect_test_stats(M, N, function_name, random, noisy)


def _collect_result(M: int, N: int, function_name: str, random: bool, noisy: bool):
    noisy_str = NORMAL_CDF_DIR if noisy else NOISELESS_DIR
    source_store = store_dir(M, N, function_name, random, noisy)
    destination = (BASE_PATH / "results") / source_store.name
    destination.mkdir(mode=0o777, parents=True, exist_ok=True)
    for gp in ("ard", "rom.optimized", "rom.reduced"):
        for sobol in (True, False):
            if sobol:
                lin_trans = linear_transformation(source_store)
                frame = data.Frame(destination / "{0}.{1}.{2}".format(noisy_str, gp, "True_Theta.csv"),  DataFrame(lin_trans))
                lin_trans = transpose(lin_trans)
                params = ("Theta.csv", "S.csv", "S1.csv")
            else:
                params = (("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv") if gp == "rom.optimized"
                          else ("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv", "test_stats.csv"))
            for param in params:
                results = None
                avg = None
                for k in range(FOLDS):
                    source = (source_store / "fold.{0:d}".format(k)) / gp
                    source = source / "sobol" if sobol else source / "kernel" if param == "lengthscale.csv" else source
                    result = data.Frame(source / param, **model.base.Model.CSV_PARAMETERS).df.copy(deep=True)
                    if param == "Theta.csv": # TODO: May not need this
                        signs = result.values @ lin_trans
                        signs = sign(diag(signs))
                        signs.shape = (signs.shape[0], 1)
                        result *= signs
                    result.insert(0, "fold", full(result.shape[0], k), True)
                    if k == 0:
                        results = result
                        avg = result / FOLDS
                    else:
                        results = concat([results, result], axis=0, ignore_index=True)
                        avg += result / FOLDS
                avg.loc[:, 'fold'] = 'mean'
                results = concat([results, avg], axis=0, ignore_index=True)
                frame = data.Frame(destination / "{0}.{1}.{2}".format(noisy_str, gp, param), results)


def _singular_values(matrix: NP.Matrix) -> NP.Covector:
    matrix = matrix @ transpose(matrix)
    return atleast_2d(sqrt(eigvalsh(matrix)))


def _analyze_theta(M: int, N: int, function_name: str, noisy: bool) -> DataFrame:
    Mu = choose_Mu(function_name)
    noisy_str = NORMAL_CDF_DIR if noisy else NOISELESS_DIR
    random_path = ((BASE_PATH / "results") / (function_name + ".{0:d}.{1:d}.random").format(N, M))
    theta_true = data.Frame(random_path / (noisy_str + ".rom.optimized.True_Theta.csv"), **model.base.Model.CSV_PARAMETERS).df.values
    theta_csv = (noisy_str + ".rom.optimized.Theta.csv")
    rom_path = ((BASE_PATH / "results") / (function_name + ".{0:d}.{1:d}.rom").format(N, M))
    theta_rom_df = data.Frame(rom_path / theta_csv, **model.base.Model.CSV_PARAMETERS).df
    theta_random_df = data.Frame(random_path / theta_csv, **model.base.Model.CSV_PARAMETERS).df
    for k in range(FOLDS):
        theta_rom = theta_rom_df.loc[theta_rom_df['fold'] == str(k)].values[:, 1:].copy().astype(float)
        theta_random = theta_random_df.loc[theta_random_df['fold'] == str(k)].values[:, 1:].transpose().copy().astype(float)
        h = theta_rom @ theta_true @ theta_random
        resultA = _singular_values(h[:Mu, :Mu])
        resultI = _singular_values(h[Mu:, Mu:])
        result = concatenate((resultA, resultI), axis=1)
        result_df = DataFrame(result)
        result_df.insert(0, "fold", full(result.shape[0], k, dtype=int), True)
        if k == 0:
            results = DataFrame(result_df)
            mean = results.copy(deep=True) / FOLDS
        else:
            results = concat([results, result_df], axis=0, ignore_index=True)
            mean += result_df / FOLDS
    mean.loc[:, 'fold'] = 'mean'
    results = concat([results, mean], axis=0, ignore_index=True)
    results.to_csv(random_path / "{0}.{1}.Theta_Analyzed.csv".format(noisy_str, "rom.optimized"))
    results.to_csv(random_path / "{0}.{1}.Theta_Analyzed.formatted.csv".format(noisy_str, "rom.optimized"), float_format='%.4f')
    return mean


def collect_results(Ms: Tuple[int], Ns: Tuple[int], function_names: Tuple[str]):
    for M in Ms:
        for N in Ns:
            for function_name in function_names:
                for noisy in (True, False):
                    for random in (True, False):
                        _collect_result(M, N, function_name, random, noisy)
                    _analyze_theta(M, N, function_name, noisy)


def summarise_results(Ms: Tuple[int], Ns: Tuple[int], function_names: Tuple[str]):
    for M in Ms:
        for function_name in function_names:
            Mu = choose_Mu(function_name)
            for random in (True, False):
                destination = (BASE_PATH / "results") / (function_name + (".random.{0:d}" if random else ".rom.{0:d}").format(M))
                destination.mkdir(mode=0o777, parents=True, exist_ok=True)
                for gp in ("ard", "rom.optimized", "rom.reduced"):
                    for sobol in (True, False):
                        if sobol:
                            params = () if gp == "rom.reduced" else ("S.csv", "S1.csv", "Theta.csv", "True_Theta.csv", "Theta_Analyzed.csv")
                        else:
                            params = (("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv") if gp == "rom.optimized"
                                      else ("lengthscale.csv", "e.csv", "f.csv", "log_likelihood.csv", "test_stats.csv"))
                        for param in params:
                            if (param == "Theta_Analyzed.csv" and ((not random) or (gp == "ard"))):
                                continue
                            is_initial = True
                            for N in Ns:
                                for noisy in (False, True):
                                    noisy_str = NORMAL_CDF_DIR if noisy else NOISELESS_DIR
                                    noise = 0.025 if noisy else 0
                                    source = (((BASE_PATH / "results")
                                               / store_dir(M, N, function_name, random, noisy).name)
                                               / "{0}.{1}.{2}".format(noisy_str, gp, param))
                                    result = data.Frame(source, **model.base.Model.CSV_PARAMETERS).df.copy(deep=True)
                                    result = (result.copy(deep=True) if param == "True_Theta.csv"
                                              else result.loc[result['fold'] == 'mean'].drop('fold', axis=1).copy(deep=True))
                                    result.insert(0, "N", full(result.shape[0], int(N/2), dtype=int), True)
                                    result.insert(0, "Noise", full(result.shape[0], noise), True)
                                    if is_initial:
                                        results = result
                                        is_initial = False
                                    else:
                                        results = concat([results, result], axis=0, ignore_index=True)
                            results.to_csv(destination / "{0}.{1}".format(gp, param), index=False)
                            results.to_csv(destination / "{0}.formatted.{1}".format(gp, param), float_format='%.4f', index=False)


def amalgamate_tests(Ms: Tuple[int], function_names: Tuple[str]):
    destination = DOC_PATH / "results"
    for M in Ms:
        for function_name in function_names:
            for random in (False, True):
                name = function_name + (".random.{0:d}" if random else ".rom.{0:d}").format(M)
                source = destination / name
                ard = read_csv(source / "ard.test_stats.csv", index_col=False).values
                reduced = read_csv(source / "rom.reduced.test_stats.csv", index_col=False).values
                amalgamated = (ard[:, :2])
                for col in range(3,6):
                    amalgamated = concatenate((amalgamated, transpose(atleast_2d(["{:4.2f}".format(100*x) for x in ard[:, col]])),
                                         transpose(atleast_2d(["{:4.2f}".format(100*x) for x in reduced[:, col]]))), axis=1)
                savetxt(fname=source / "amalgamated.test_stats.csv", X=amalgamated,
                        fmt=("%4.3f","%d", "%s", "%s", "%s", "%s", "%s", "%s"), delimiter=",")


def prepare_tables(Ms: Tuple[int], function_names: Tuple[str], noise_selectors: Tuple[Tuple[bool]]):
    destination = DOC_PATH / "results"
    for M in Ms:
        for function_name in function_names:
                name = function_name + ".random.{0:d}".format(M)
                source = destination / name


def copy_results(Ms: Tuple[int], function_names: Tuple[str]):
    destination = DOC_PATH / "results"
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(mode=0o777, parents=True, exist_ok=True)
    for M in Ms:
        for function_name in function_names:
            for random in (True, False):
                name = function_name + (".random.{0:d}" if random else ".rom.{0:d}").format(M)
                source = (BASE_PATH / "results") / name
                shutil.copytree(source, destination / name)


if __name__ == '__main__':
    start_time = time()
    intermediate_time = time()
    run_tests(Ms=(5,), Ns=(200, 400, 800, 1600, 3200), function_names=("sin.1", "sin.2", "ishigami", "sobol_g"))
    print("tests finished in {0:.1f} minutes.".format((time() - intermediate_time) / 60))
    intermediate_time = time()
    collect_tests(Ms=(5,), Ns=(200, 400, 800, 1600, 3200), function_names=("sin.1", "sin.2", "ishigami", "sobol_g"))
    print("collecting tests finished in {0:.1f} minutes.".format((time() - intermediate_time) / 60))
    intermediate_time = time()
    collect_results(Ms=(5,), Ns=(200, 400, 800, 1600, 3200), function_names=("sin.1", "sin.2", "ishigami", "sobol_g"))
    print("collecting results finished in {0:.1f} minutes.".format((time() - intermediate_time) / 60))
    intermediate_time = time()
    summarise_results(Ms=(5,), Ns=(200, 400, 800, 1600, 3200), function_names=("sin.1", "sin.2", "ishigami", "sobol_g"))
    print("summarising results finished in {0:.1f} minutes.".format((time() - intermediate_time) / 60))
    copy_results(Ms=(5,), function_names=("sin.1", "sin.2", "ishigami", "sobol_g"))
    amalgamate_tests(Ms=(5,), function_names=("sin.1", "sin.2", "ishigami", "sobol_g"))
    print("main finished in {0:.1f} minutes.".format((time() - start_time) / 60))

