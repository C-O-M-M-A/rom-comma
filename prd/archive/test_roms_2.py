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
from numpy import zeros, eye, pi, full, atleast_2d, savetxt, transpose, einsum, loadtxt
from pandas import MultiIndex, DataFrame, concat
from pathlib import Path
from scipy.stats import ortho_group

EFFECTIVELY_ZERO = 1.0E-64
BASE_PATH = Path('X:\\comma_group1\\Rom\\dat\\TestFunctions\\ROM\\3')


def scalar_function_of_normal(store_name: str, N: int, M: int, X_std: float, noise_std: float, CDF_scale: NP.Array=None, CDF_loc: NP.Array=None,
                        pre_function_with_parameters: function.CallableWithParameters = None,
                       function_with_parameters: function.CallableWithParameters = None) -> data.Store:
    X_marginal = distribution.Univariate('norm', loc=0, scale=X_std)
    X_dist = distribution.Multivariate.Independent(M=M, marginals=X_marginal)
    noise_dist = (distribution.Multivariate.Normal(mean=zeros(1, dtype=float), covariance=noise_std ** 2 * eye(1, dtype=float))
                  if noise_std > EFFECTIVELY_ZERO else None)
    return function.sample(store_dir=store_name, N=N, X_distribution=X_dist,
                               X_sample_design=distribution.SampleDesign.LATIN_HYPERCUBE, CDF_scale=CDF_scale,
                               CDF_loc=CDF_loc, pre_function_with_parameters=pre_function_with_parameters,
                               functions_with_parameters=function_with_parameters,
                               noise_distribution=noise_dist, noise_sample_design=distribution.SampleDesign.LATIN_HYPERCUBE)


def run_roms(test_fuction: str, N: int, noise_std: float, random: bool, M: int = 5, K: int = 2 ):
    name = 'rbf'
    gp_optimizer_options = {'optimizer': 'bfgs', 'max_iters': 5000, 'gtol': 1E-16}
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, 1), 2.5**(M/5), dtype=float))
    parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-6, e=0.003)
    CDF_scale = 2 * pi
    CDF_loc = pi
    if test_fuction == 'sin.1':
        function_with_parameters = function.CallableWithParameters(function.ishigami, parameters={'a': 0.0, 'b': 0.0})
    elif test_fuction == 'sin.2':
        function_with_parameters = function.CallableWithParameters(function.ishigami, parameters={'a': 2.0, 'b': 0.0})
    elif test_fuction == 'ishigami':
        function_with_parameters = function.callable_with_parameters(function.ishigami)
    else:
        CDF_scale = 1.0
        CDF_loc = 0.0
        function_with_parameters = function.CallableWithParameters(function.sobol_g,
                                                                   parameters={'m_very_important': 2, 'm_important': 3, 'm_unimportant': 4})
    store_name = test_fuction + '.{0:d}.{1:.3f}.{2:d}'.format(M, noise_std, N)
    if random:
        lin_trans = ortho_group.rvs(M)
        pre_function_with_parameters = function.CallableWithParameters(function=function.linear, parameters={'matrix': lin_trans})
        store_name += '.random'
    else:
        pre_function_with_parameters = None
        store_name += '.rom'
    store_name = BASE_PATH / store_name
    store = scalar_function_of_normal(store_name=store_name, N=N, M=M, X_std=1.0, noise_std=noise_std, CDF_scale=CDF_scale, CDF_loc=CDF_loc,
                                      pre_function_with_parameters=pre_function_with_parameters,
                                      function_with_parameters=function_with_parameters)
    savetxt(store.dir / "InverseRotation.csv", transpose(lin_trans))
    data.Fold.into_K_folds(parent=store, K=K, shuffled_before_folding=False, standard=data.Store.Standard.mean_and_std,
                           replace_empty_test_with_data_=True)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=parameters, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=None, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options, make_ard=True)
    sobol_options = {'semi_norm': model.base.Sobol.SemiNorm.DEFAULT_META, 'N_exploit': 3, 'N_explore': 4096, 'options': {'gtol': 1.0E-16}}
    rom_options = {'iterations': 1, 'guess_identity_after_iteration': -1, 'sobol_optimizer_options': sobol_options,
                                 'gp_initializer': model.base.ROM.GP_Initializer.RBF,
                                 'gp_optimizer_options': gp_optimizer_options}
    model.run.ROMs(module=model.run.Module.GPY_, name='rom', store=store, source_gp_name=name, Mu=-1, Mx=-1, optimizer_options=rom_options,
                   rbf_parameters=parameters)


def test_random(test_fuction: str, N: int, noise_std: float, M: int = 5, K: int = 2 ):
    random = True
    name = 'rbf'
    gp_optimizer_options = {'optimizer': 'bfgs', 'max_iters': 5000, 'gtol': 1E-16}
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, 1), 2.5**(M/5), dtype=float))
    parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-6, e=0.003)
    CDF_scale = 2 * pi
    CDF_loc = pi
    if test_fuction == 'sin.1':
        function_with_parameters = function.CallableWithParameters(function.ishigami, parameters={'a': 0.0, 'b': 0.0})
    elif test_fuction == 'sin.2':
        function_with_parameters = function.CallableWithParameters(function.ishigami, parameters={'a': 2.0, 'b': 0.0})
    elif test_fuction == 'ishigami':
        function_with_parameters = function.callable_with_parameters(function.ishigami)
    else:
        CDF_scale = 1.0
        CDF_loc = 0.0
        function_with_parameters = function.CallableWithParameters(function.sobol_g,
                                                                   parameters={'m_very_important': 2, 'm_important': 3, 'm_unimportant': 4})
    store_name = test_fuction + '.{0:d}.{1:.3f}.{2:d}'.format(M, noise_std, N)
    if random:
        lin_trans = ortho_group.rvs(M)
        pre_function_with_parameters = function.CallableWithParameters(function=function.linear, parameters={'matrix': lin_trans})
        store_name += '.random'
    else:
        pre_function_with_parameters = None
        store_name += '.rom'
    store_name = BASE_PATH / store_name
    store = scalar_function_of_normal(store_name=store_name, N=N, M=M, X_std=1.0, noise_std=noise_std, CDF_scale=CDF_scale, CDF_loc=CDF_loc,
                                      pre_function_with_parameters=pre_function_with_parameters,
                                      function_with_parameters=function_with_parameters)
    # lin_trans = transpose(lin_trans)
    savetxt(store.dir / "InverseRotation.csv", lin_trans)
    data.Fold.into_K_folds(parent=store, K=K, shuffled_before_folding=False, standard=data.Store.Standard.mean_and_std,
                           replace_empty_test_with_data_=True)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=parameters, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=None, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options, make_ard=True)
    sobol_options = {'semi_norm': model.base.Sobol.SemiNorm.DEFAULT_META, 'N_exploit': 3, 'N_explore': 4096, 'options': {'gtol': 1.0E-16}}
    for k in range(K):
        sobol = model.gpy_.Sobol.from_GP(data.Fold(store, k), 'rbf.ard', 'rbf.ard.derotated')
        for sobol.m in reversed(range(M)):
            sobol.Theta_old = lin_trans
        sobol.write_parameters(sobol.Parameters(Mu=sobol.Mu, Theta=sobol.Theta, D=sobol.Tensor3AsMatrix(sobol.D), S1=None,
                                                S=sobol.Tensor3AsMatrix(sobol.S)))


def replace_X_with_U(fold: data.Fold, Theta: NP.Matrix):
    """ Replace X with its rotated/reordered version U."""
    column_headings = MultiIndex.from_product(((fold.meta['data']['X_heading'],), ("u{:d}".format(i) for i in range(fold.M))))
    X = DataFrame(einsum('MK, NK -> NM', Theta, fold.X, optimize=True, dtype=float),
                  columns=column_headings, index=fold.X.index)
    test_X = DataFrame(einsum('MK, NK -> NM', Theta, fold.test_X, optimize=True, dtype=float),
                       columns=column_headings, index=fold.test_X.index)
    fold.data.df = concat((X, fold.data.df[[fold.meta['data']['Y_heading']]].copy(deep=True)), axis='columns')
    fold.data.write()
    fold.test.df = concat((test_X, fold.test.df[[fold.meta['data']['Y_heading']]].copy(deep=True)), axis='columns')
    fold.test.write()
    fold.meta_data_update()

def test_random2(test_fuction: str, N: int, noise_std: float, M: int = 5, K: int = 2 ):
    random = True
    name = 'rbf'
    gp_optimizer_options = {'optimizer': 'bfgs', 'max_iters': 5000, 'gtol': 1E-16}
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, 1), 2.5**(M/5), dtype=float))
    parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-6, e=0.003)
    CDF_scale = 2 * pi
    CDF_loc = pi
    if test_fuction == 'sin.1':
        function_with_parameters = function.CallableWithParameters(function.ishigami, parameters={'a': 0.0, 'b': 0.0})
    elif test_fuction == 'sin.2':
        function_with_parameters = function.CallableWithParameters(function.ishigami, parameters={'a': 2.0, 'b': 0.0})
    elif test_fuction == 'ishigami':
        function_with_parameters = function.callable_with_parameters(function.ishigami)
    else:
        CDF_scale = 1.0
        CDF_loc = 0.0
        function_with_parameters = function.CallableWithParameters(function.sobol_g,
                                                                   parameters={'m_very_important': 2, 'm_important': 3, 'm_unimportant': 4})
    store_name = test_fuction + '.{0:d}.{1:.3f}.{2:d}'.format(M, noise_std, N)
    if random:
        lin_trans = ortho_group.rvs(M)
        pre_function_with_parameters = function.CallableWithParameters(function=function.linear, parameters={'matrix': lin_trans})
        store_name += '.random'
    else:
        pre_function_with_parameters = None
        store_name += '.rom'
    store_name = BASE_PATH / store_name
    store = scalar_function_of_normal(store_name=store_name, N=N, M=M, X_std=1.0, noise_std=noise_std, CDF_scale=CDF_scale, CDF_loc=CDF_loc,
                                      pre_function_with_parameters=pre_function_with_parameters,
                                      function_with_parameters=function_with_parameters)
    savetxt(store.dir / "InverseRotation.csv", lin_trans)
    data.Fold.into_K_folds(parent=store, K=K, shuffled_before_folding=False, standard=data.Store.Standard.mean_and_std,
                           replace_empty_test_with_data_=True)
    for k in range(K):
        fold = data.Fold(store, k)
        replace_X_with_U(fold, lin_trans)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=parameters, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=None, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options, make_ard=True)
    sobol_options = {'semi_norm': model.base.Sobol.SemiNorm.DEFAULT_META, 'N_exploit': 3, 'N_explore': 4096, 'options': {'gtol': 1.0E-16}}

def test_random3(test_fuction: str, N: int, noise_std: float, M: int = 5, K: int = 2 ):
    random = True
    name = 'derotated.rbf'
    gp_optimizer_options = {'optimizer': 'bfgs', 'max_iters': 5000, 'gtol': 1E-16}
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, 1), 2.5**(M/5), dtype=float))
    parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-6, e=0.003)
    store_name = test_fuction + '.{0:d}.{1:.3f}.{2:d}'.format(M, noise_std, N)
    store_name += '.random'
    store_name = BASE_PATH / store_name
    store = data.Store(store_name)
    lin_trans = loadtxt(store.dir / "InverseRotation.csv")
    for k in range(K):
        fold = data.Fold(store, k)
        replace_X_with_U(fold, transpose(lin_trans))
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=parameters, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=None, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options, make_ard=True)
    sobol_options = {'semi_norm': model.base.Sobol.SemiNorm.DEFAULT_META, 'N_exploit': 3, 'N_explore': 4096, 'options': {'gtol': 1.0E-16}}

if __name__ == '__main__':
    for M in (5, ):
        for N in (400, ):
            for noise_std in (0.01, ):
                for random in (True, ):
                    test_random3("sin.2", N, noise_std, M)
