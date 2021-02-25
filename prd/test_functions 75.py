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
from romcomma.typing_ import NP
from numpy import zeros, eye, pi, full, atleast_2d
from pathlib import Path
from scipy.stats import ortho_group

EFFECTIVELY_ZERO = 1.0E-64
BASE_PATH = Path('X:\\comma_group1\\Rom\\dat\\TestFunctions\\Scalar.3')
NOISELESS_DIR = 'Noiseless'
NORMAL_DIR = 'Normal'
UNIFORM_DIR = 'Uniform'
NORMAL_CDF_DIR = 'NormalCDF'


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


def run_roms(M: int, N: int, K:int, ishigami: bool, random: bool, noisy: bool):
    name = 'ard'
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, M), 0.2, dtype=float))
    if ishigami:
        store_name = 'sin.2.'
        CDF_scale = 2 * pi
        CDF_loc = pi
        function_with_parameters = function.CallableWithParameters(function=function.ishigami, parameters={'a': 2.0, 'b': 0})
    else:
        store_name = 'sin.1.'
        CDF_scale = 2 * pi
        CDF_loc = pi
        function_with_parameters = function.CallableWithParameters(function=function.ishigami, parameters={'a': 0, 'b': 0})
    store_name = store_name + '{0:d}.{1:d}'.format(N, M)
    if random:
        pre_function_with_parameters = function.CallableWithParameters(function=function.linear, parameters={'matrix': ortho_group.rvs(M)})
        store_name += '.random'
    else:
        pre_function_with_parameters = None
        store_name += '.rom'
    if noisy:
        noise_std = 0.001
        parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-4, e=0.001)
    else:
        noise_std = 0
        parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-5, e=1E-10)
    store = scalar_function_of_normal(store_name=store_name, N=N, M=M, X_std=1.0, noise_std=noise_std, CDF_scale=CDF_scale, CDF_loc=CDF_loc,
                                      pre_function_with_parameters=pre_function_with_parameters,
                                      function_with_parameters=function_with_parameters)
    data.Fold.into_K_folds(parent=store, K=K, shuffled_before_folding=False, standard=data.Store.Standard.mean_and_std,
                           replace_empty_test_with_data_=True)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=parameters, optimize=True, test=True, sobol=True)
    sobol_options = {'semi_norm': model.base.Sobol.SemiNorm.DEFAULT_META, 'N_exploit': 3, 'N_explore': 4096, 'options': {'gtol': 1.0E-16}}
    rom_options = {'iterations': 4, 'guess_identity_after_iteration': 2, 'sobol_optimizer_options': sobol_options,
                                 'gp_initializer': model.base.ROM.GP_Initializer.CURRENT_WITH_GUESSED_LENGTHSCALE,
                                 'gp_optimizer_options': model.run.Module.GPY_.value.GP.DEFAULT_OPTIMIZER_OPTIONS}
    model.run.ROMs(module=model.run.Module.GPY_, name='rom', store=store, source_gp_name='ard', Mu=-1, Mx=-1, optimizer_options=rom_options)


if __name__ == '__main__':
    for M in (5, ):
        for noisy in (True, ):
            for N in (200, 400, 800, 1600, 3200):
                for ishigami in (True,):
                    for random in (True,):
                            run_roms(M, N, 2, ishigami, random, noisy)
