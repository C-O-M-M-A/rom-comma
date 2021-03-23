# BSD 3-Clause License
#
# Copyright (c) 2019-2021, Robert A. Milton
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Run this module first thing, to test your installation of romcomma. """

from romcomma import model
from romcomma.data import Fold, Store
from romcomma.function import FunctionWithParameters, functions_of_normal, linear
from numpy import eye, savetxt, transpose, full
from pathlib import Path
from scipy.stats import ortho_group

BASE_PATH = Path('C:\\Users\\fc1ram\\Documents\\Rom\\dat\\SoftwareTest\\0.0')


# noinspection PyShadowingNames
def run_gps(name, function_name: str, N: int, noise_std: float, random: bool, M: int = 5, K: int = 2):
    store_dir = f'{function_name}.{M:d}.{noise_std:.3f}.{N:d}'
    if random:
        lin_trans = ortho_group.rvs(M)
        input_transform = FunctionWithParameters(function_=linear, parameters_={'matrix': lin_trans})
        store_dir += '.random'
    else:
        lin_trans = eye(M)
        input_transform = None
        store_dir += '.rom'
    store_dir = BASE_PATH / store_dir
    CDF_loc, CDF_scale, functions = FunctionWithParameters.default(function_name)
    store = functions_of_normal(store_dir=store_dir, N=N, M=M, CDF_loc=CDF_loc, CDF_scale=CDF_scale,
                                input_transform=input_transform, functions=functions, noise_std=noise_std)
    savetxt(store.dir / 'InverseRotation.csv', transpose(lin_trans))
    Fold.into_K_folds(parent=store, K=K, shuffled_before_folding=False, standard=Store.Standard.mean_and_std, replace_empty_test_with_data_=True)
    gp_optimizer_options = {'optimizer': 'bfgs', 'max_iters': 5000, 'gtol': 1E-16}
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, 1), 2.5 ** (M / 5), dtype=float))
    # noinspection PyProtectedMember
    gp_parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-6, e=0.003)
    # model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=gp_parameters, optimize=True, test=True, sobol=False,
    #               optimizer_options=gp_optimizer_options, make_ard=False)
    # model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=None, optimize=True, test=True, sobol=True,
    #               optimizer_options=gp_optimizer_options, make_ard=True)


if __name__ == '__main__':
    for N in (800,):
        for noise_std in (0,):
            for random in (False,):
                run_gps('initial', 'sin.1', N, noise_std, random, M=1)
