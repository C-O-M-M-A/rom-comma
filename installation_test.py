#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2022 Robert A. Milton. All rights reserved.
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

from romcomma.base.definitions import *
from romcomma import run
from romcomma.test.utilities import sample


BASE_PATH = Path('./installation_test')


if __name__ == '__main__':
    with run.Context('Test', float='float64', device='CPU'):
        for N in (500,):
            for M in (5,):
                for noise_magnitude in (0.1,):
                    for is_rotated in (False, ):
                        with run.TimingOneLiner(f'sample generation for N={N}, noise={noise_magnitude}'):
                            repo = sample(BASE_PATH, ['sin.1', 'sin.1'], N, M, K=1,
                                          noise_magnitude=noise_magnitude, is_noise_diagonal=False, is_noise_variance_stochastic=False)
                        with run.Timing(f'Gaussian Process Regression for N={N}, noise={noise_magnitude}'):
                            run.gpr(name='test', repo=repo, is_read=None, is_isotropic=False, is_independent=None, optimize=True, test=True)
                        with run.Timing(f'Global Sensitivity Analysis for N={N}, noise={noise_magnitude}'):
                            run.gsa(name='test', repo=repo, is_independent=True, is_isotropic=False)
                            run.gsa(name='test', repo=repo, is_independent=False, is_isotropic=False)
