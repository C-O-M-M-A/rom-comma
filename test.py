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

from romcomma.typing_ import *
from romcomma import model
from romcomma.data import Fold, Store
from romcomma.function import Matrix, FunctionWithParameters, functions_of_normal
from numpy import eye, savetxt, transpose, full, atleast_2d
from pathlib import Path
from scipy.stats import ortho_group

BASE_PATH = Path('C:\\Users\\fc1ram\\Documents\\Rom\\dat\\SoftwareTest\\0.0')


# noinspection PyShadowingNames
def run_gps(name, function_name: Sequence[str], N: int, noise_std: float, random: bool, M: int = 5, K: int = 2):
    if isinstance(function_name, str):
        function_name = [function_name]
    store_dir = '.'.join(function_name) + f'.{M:d}.{noise_std:.3f}.{N:d}'
    if random:
        lin_trans = ortho_group.rvs(M)
        input_transform = FunctionWithParameters(function_= Matrix.multiply, parameters_={'matrix': lin_trans})
        store_dir += '.random'
    else:
        lin_trans = eye(M)
        input_transform = None
        store_dir += '.rom'
    store_dir = BASE_PATH / store_dir
    CDF_loc, CDF_scale, functions = FunctionWithParameters.default(function_name)
    store = functions_of_normal(store_dir=store_dir, N=N, M=M, CDF_loc=CDF_loc, CDF_scale=CDF_scale,
                                input_transform=input_transform, functions=functions, noise_std=noise_std)
    savetxt(store.folder / 'InverseRotation.csv', transpose(lin_trans))
    Fold.into_K_folds(parent=store, K=K, shuffled_before_folding=False, standard=Store.Standard.mean_and_std, replace_empty_test_with_data_=True)
    model.run.gps(name=name, store=store, M=M, is_read=False, is_isotropic=False, is_independent=True, kernel_parameters=None, parameters=None,
                  optimize=True, test=True)


if __name__ == '__main__':
    with model.run.Running('Test'):
        for N in (800,):
            for noise_std in (0,):
                for random in (False,):
                    run_gps('initial', ['sin.1', 'sin.2'], N, noise_std, random, M=2)
