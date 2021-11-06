#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2021 Robert A. Milton. All rights reserved.
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

""" Run this module first thing, to test_data your installation of romcomma. """

from __future__ import annotations

from romcomma.typing_ import *
from romcomma import run
from romcomma.data import Fold, Store
from romcomma.test import functions, sampling
from numpy import eye, savetxt, transpose
from pathlib import Path
from scipy.stats import ortho_group

BASE_PATH = Path('C:\\Users\\fc1ram\\Documents\\Rom\\dat\\SoftwareTest\\0.0')


# noinspection PyShadowingNames
def run_gps(name, function_names: Sequence[str], N: int, noise_variance: [float], random: bool, M: int = 5, K: int = 2):
    if isinstance(function_names, str):
        function_name = [function_names]
    f = tuple((functions.FunctionWithMeta.DEFAULTS[function_name] for function_name in function_names))
    store_folder = '.'.join(function_names) + f'.{M:d}.{sum(noise_variance)/len(noise_variance):.3f}.{N:d}'
    if random:
        rotation = ortho_group.rvs(M)
        store_folder += '.random'
    else:
        rotation = eye(M)
        store_folder += '.rom'
    store_folder = BASE_PATH / store_folder
    store = functions.sample(f, N, M, noise_variance, store_folder)
    store.into_K_folds(K=2)
#     run.gps(name=name, store=store, M=M, is_read=False, is_isotropic=False, is_independent=True, kernel_parameters=None, parameters=None,
#                   optimize=True, test=True)


if __name__ == '__main__':
    with run.Context('Test'):
        for N in (800,):
            for noise_variance in (0.3,):
                for random in (True,):
                    for M in (5,):
                        run_gps('initial', ['sin.1', 'sin.2'], N, [noise_variance] * 2, random, M)
