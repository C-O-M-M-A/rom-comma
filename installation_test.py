#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2023 Robert A. Milton. All rights reserved.
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

""" Run this module first thing, to test your installation of romcomma. """

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma import run, test



BASE_FOLDER = Path('installation_test')


if __name__ == '__main__':
    with run.Context('Test', float='float64', device='CPU'):
        function_vector = test.functions.ISHIGAMI.subVector('ishigami', ['standard', 'inflated'])
        for N in (500,):
            for M in (5,):
                for noise_magnitude in (0.1,):
                    with run.TimingOneLiner(f'sample generation for N={N}, noise={noise_magnitude}'):
                        noise_variance = test.sample.GaussianNoise.Variance(len(function_vector), noise_magnitude, False, False)
                        sample = test.sample.Function(BASE_FOLDER, test.sample.DOE.latin_hypercube, function_vector, N, M, noise_variance, True)
                        repo = sample.into_K_folds(K=1).rotate_folds(None).repo
                    with run.Timing(f'Gaussian Process Regression for N={N}, noise={noise_magnitude}'):
                        run.gpr(name='test', repo=repo, is_read=None, is_independent=None, is_isotropic=False, optimize=True, test=True)
                    with run.Timing(f'Global Sensitivity Analysis for N={N}, noise={noise_magnitude}'):
                        run.GSA(name='test', repo=repo, is_independent=None, is_isotropic=False)
