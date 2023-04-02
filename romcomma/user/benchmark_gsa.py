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

""" **Example scripts for users to cannibalize** """

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma import gsa
from romcomma.run import context, function, results, sample, summarised

BASE_FOLDER = Path('C:/Users/fc1ram/Documents/Research/rom-comma/dat/SoftwareTest/1.1.3')     #: The base folder to house all data repositories.

if __name__ == '__main__':
    function_vector = function.ALL
    models = ['diag.i.a', 'diag.d.a']
    overwrite_existing = True
    ignore_exceptions = False
    kinds = gsa.run.calculation.ALL_KINDS
    is_error_calculated = True
    is_T_partial = False
    with context.Environment('Test', device='GPU'):
        kind_names = [kind.name.lower() for kind in kinds]
        for M in (10, 7, 13, 18):
            for N in (3E4, 1E4, 7E3, 2E3, 1680, 1280, 960, 720, 520, 240, 200, 160, 128, 60, 40, 20):
                for noise_magnitude in (0.5, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.0025, 1.0, 0.75, 10.0, 5.0, 2.0):
                    for is_noise_diagonal in (False, True):
                        with context.TimingOneLiner(f'M={M}, N={N}, noise={noise_magnitude} is_noise_diagonal={is_noise_diagonal} \n'):
                            noise_variance = sample.GaussianNoise.Variance(len(function_vector), noise_magnitude, is_noise_diagonal, False)
                            if overwrite_existing:
                                repo = sample.Function(BASE_FOLDER, sample.DOE.latin_hypercube, function_vector, N, M, noise_variance, True)
                                repo = repo.into_K_folds(K=2).rotate_folds(None).repo
                                summarised.gpr(name='diag', repo=repo, is_read=None, is_covariant=None, is_isotropic=False,
                                              ignore_exceptions=ignore_exceptions, optimize=True, test=True)
                            else:
                                repo = sample.Function(BASE_FOLDER, sample.DOE.latin_hypercube, function_vector, N, M, noise_variance, False).repo
                            run.Collect({'test': {'header': [0, 1]}, 'test_summary': {'header': [0, 1], 'index_col': 0}},
                                          {repo.folder/model: {'model': model} for model in models}, ignore_exceptions).over_folders(repo.folder/'gpr', True)
                            run.Collect({'variance': {}, 'log_marginal': {}}, {f'{repo.folder / model}/likelihood': {'model': model} for model in models},
                                          ignore_exceptions).over_folders((repo.folder/'gpr')/'likelihood', True)
                            run.Collect({'variance': {}, 'lengthscales': {}}, {f'{repo.folder / model}/kernel': {'model': model} for model in models},
                                          ignore_exceptions).over_folders((repo.folder/'gpr')/'kernel', True)
                            run.GSA('diag', repo, is_covariant=None, is_isotropic=False, kinds=kinds, is_error_calculated=is_error_calculated,
                                    ignore_exceptions=ignore_exceptions, is_T_partial=is_T_partial)
                            run.Collect({'S': {}, 'V': {}} | ({'T': {}, 'W': {}} if is_error_calculated else {}),
                                          {f'{repo.folder/model}/gsa/{kind_name}': {'model': model, 'kind': kind_name}
                                           for kind_name in kind_names for model in models},
                                          ignore_exceptions).over_folders((repo.folder/'gsa'), True)

