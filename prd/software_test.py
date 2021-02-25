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
from numpy import zeros, eye, pi, full, atleast_2d, savetxt, transpose, einsum, loadtxt, array
from pandas import MultiIndex, DataFrame, concat
from pathlib import Path
from scipy.stats import ortho_group

EFFECTIVELY_ZERO = 1.0E-64
BASE_PATH = Path('X:\\comma_group1\\Rom\\dat\\SoftwareTest\\1.0')


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


def run_gps(name, test_fuction: str, N: int, noise_std: float, random: bool, M: int = 5, K: int = 2):
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
        lin_trans = eye(M)
        pre_function_with_parameters = None
        store_name += '.rom'
    store_name = BASE_PATH / store_name
    store = scalar_function_of_normal(store_name=store_name, N=N, M=M, X_std=1.0, noise_std=noise_std, CDF_scale=CDF_scale, CDF_loc=CDF_loc,
                                      pre_function_with_parameters=pre_function_with_parameters,
                                      function_with_parameters=function_with_parameters)
    savetxt(store.dir / "InverseRotation.csv", transpose(lin_trans))
    data.Fold.into_K_folds(parent=store, K=K, shuffled_before_folding=False, standard=data.Store.Standard.none,
                           replace_empty_test_with_data_=True)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=parameters, optimize=True, test=True, sobol=False,
                  optimizer_options=gp_optimizer_options, make_ard=False)
    model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=None, optimize=True, test=True, sobol=True,
                  optimizer_options=gp_optimizer_options, make_ard=True)


def playtest(name, test_fuction: str, N: int, noise_std: float, random: bool, M: int = 5, K: int = 2):
    # gp_optimizer_options = {'optimizer': 'bfgs', 'max_iters': 5000, 'gtol': 1E-16}
    # kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, 1), 2.5**(M/5), dtype=float))
    # parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-6, e=0.003)
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
        lin_trans = eye(M)
        pre_function_with_parameters = None
        store_name += '.rom'
    store_name = BASE_PATH / store_name
    store = scalar_function_of_normal(store_name=store_name, N=N, M=M, X_std=1.0, noise_std=noise_std, CDF_scale=CDF_scale, CDF_loc=CDF_loc,
                                      pre_function_with_parameters=pre_function_with_parameters,
                                      function_with_parameters=function_with_parameters)
    column_headings = store.data.df.columns
    bummer = column_headings.levels[1].to_numpy()
    reorder = array([1,0,4,3,2])
    bummer[:reorder.shape[0]] = bummer[reorder]
    column_headings = column_headings.reindex(bummer, level=1)
    store.data.df.reindex(columns=column_headings[0], copy=False)
    Y = store.Y
    # column_headings = MultiIndex.from_product(((self._gp.fold.meta['data']['X_heading'],), ("u{:d}".format(i) for i in range(self.Mu))))
    # X = DataFrame(einsum('MK, NK -> NM', self.Theta_old, self._gp.X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT),
    #               columns=column_headings, index=self._gp.fold.X.index)
    # test_X = DataFrame(einsum('MK, NK -> NM', self.Theta_old, self._gp.fold.test_X, optimize=True, dtype=float, order=self.MEMORY_LAYOUT),
    #                    columns=column_headings, index=self._gp.fold.test_X.index)
    # self._gp.fold.data.df = concat((X, self._gp.fold.data.df[[self._gp.fold.meta['data']['Y_heading']]].copy(deep=True)), axis='columns')
    # self._gp.fold.data.write()

    # savetxt(store.dir / "InverseRotation.csv", transpose(lin_trans))
    # data.Fold.into_K_folds(parent=store, K=K, shuffled_before_folding=False, standard=data.Store.Standard.mean_and_std,
    #                        replace_empty_test_with_data_=True)
    # model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=parameters, optimize=True, test=True, sobol=True,
    #               optimizer_options=gp_optimizer_options)
    # model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=None, optimize=True, test=True, sobol=True,
    #               optimizer_options=gp_optimizer_options, make_ard=True)


if __name__ == '__main__':
    for N in (800, ):
        for noise_std in (0.00001, ):
            for random in (False, ):
                run_gps("initial", "sin.1", N, noise_std, random, M=5)
