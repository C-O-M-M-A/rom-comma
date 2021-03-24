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
from numpy import zeros, eye, pi, full, atleast_2d, array
from pathlib import Path
from json import load
from scipy.stats import ortho_group

EFFECTIVELY_ZERO = 1.0E-64
BASE_PATH = Path('X:\\comma_group1\\Rom\\dat\\TestFunctions\\Scalar')
NOISELESS_DIR = 'Noiseless'
NORMAL_DIR = 'Normal'
UNIFORM_DIR = 'Uniform'
NORMAL_CDF_DIR = 'NormalCDF'


def scalar_function_of_normal(store_name: str, N: int, M: int, X_std: float, noise_std: float, CDF_scale: NP.Array=None, CDF_loc: NP.Array=None,
                        pre_function_with_parameters: function.CallableWithParameters = None,
                       function_with_parameters: function.CallableWithParameters = None) -> data.Store:
    X_marginal = distribution.Univariate('norm', loc=0, scale=X_std)
    X_dist = distribution.Multivariate.Independent(M=M, marginals=X_marginal)
    noise_dist = (distribution.Multivariate.Normal(mean=zeros(1, dtype=float), covariance=noise_std ** 2 * eye(1, dtype=float))
                  if noise_std > EFFECTIVELY_ZERO else None)
    df, meta = function.sample(N=N, X_distribution=X_dist, X_sample_design=distribution.SampleDesign.LATIN_HYPERCUBE, CDF_scale=CDF_scale,
                               CDF_loc=CDF_loc, pre_function_with_parameters=pre_function_with_parameters,
                               functions_with_parameters=function_with_parameters,
                               noise_distribution=noise_dist, noise_sample_design=distribution.SampleDesign.LATIN_HYPERCUBE)
    if noise_dist is None:
        dir_ = BASE_PATH / NOISELESS_DIR / store_name
    elif CDF_scale is None:
        dir_ = BASE_PATH / NORMAL_DIR / store_name
    else:
        dir_ = BASE_PATH / NORMAL_CDF_DIR / store_name
    return data.Store.from_df(dir_=dir_, df=df, meta=meta)


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
        return eye(meta['data']['M'], dtype=float)


if __name__ == '__main__':
    store = data.Store((BASE_PATH / NOISELESS_DIR) / "sin.u1.5000.5.random", data.Store.InitMode.READ_META_ONLY)
    fold = data.Fold(store, 0)
    rom = model.gpy_.ROM.from_ROM(fold=fold, name='rom', suffix='.test.full')
    model_theta = rom.sobol.parameters_read.Theta
    data_theta = function.linear_matrix_from_meta(store)
    print(model_theta)
    print(data_theta)


"""
def rename(dir_: Path):
    for p in dir_.iterdir():
        if p.is_dir():
            if p.name == "rom..optimized":
                p.replace(p.parent / "rom.optimized")
            else:
                rename(p)
"""
"""
if __name__ == '__main__':
    rename(BASE_PATH)
"""