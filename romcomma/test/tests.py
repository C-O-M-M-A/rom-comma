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

""" Contains developer tests of romcomma. """

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma import run, gsa
from romcomma.test import sample, functions

BASE_FOLDER = Path('C:/Users/fc1ram/Documents/Research/dat/SoftwareTest/1.1.3')


if __name__ == '__main__':
    function_vector = functions.OAKLEY2004
    models = ['diag.i.a', 'diag.d.a']
    overwrite_existing = False
    ignore_exceptions = False
    kinds = gsa.run.calculation.ALL_KINDS
    is_error_calculated = True
    with run.Context('Test', device='CPU'):
        kind_names = [kind.name.lower() for kind in kinds]
        for N in (200,):
            for M in (10,):
                for noise_magnitude in (0.2,):
                    for is_noise_independent in (False,):
                        with run.TimingOneLiner(f'M={M}, N={N}, noise={noise_magnitude} \n'):
                            noise_variance = sample.GaussianNoise.Variance(len(function_vector), noise_magnitude, False, False)
                            if overwrite_existing:
                                repo = sample.Function(BASE_FOLDER, sample.DOE.latin_hypercube, function_vector, N, M, noise_variance, True)
                                repo = repo.into_K_folds(K=1).rotate_folds(None).repo
                                run.gpr(name='diag', repo=repo, is_read=None, is_independent=None, is_isotropic=False, ignore_exceptions=ignore_exceptions,
                                        optimize=True, test=True)
                            else:
                                repo = sample.Function(BASE_FOLDER, sample.DOE.latin_hypercube, function_vector, N, M, noise_variance, False).repo
                            run.Aggregate({'test': {'header': [0, 1]}, 'test_summary': {'header': [0, 1], 'index_col': 0}}, {repo.folder/model: {'model': model} for model in models},
                                          ignore_exceptions).over_folders(repo.folder/'gpr', True)
                            run.Aggregate({'variance': {}, 'log_marginal': {}}, {f'{repo.folder/model}/likelihood': {'model': model} for model in models},
                                          ignore_exceptions).over_folders((repo.folder/'gpr')/'likelihood', True)
                            run.Aggregate({'variance': {}, 'lengthscales': {}}, {f'{repo.folder/model}/kernel': {'model': model} for model in models},
                                          ignore_exceptions).over_folders((repo.folder/'gpr')/'kernel', True)
                            run.GSA('diag', repo, is_independent=None, is_isotropic=False, kinds=kinds, is_error_calculated=is_error_calculated,
                                    ignore_exceptions=ignore_exceptions, is_T_partial=False)
                            run.Aggregate({'S': {}, 'V': {}} | ({'T': {}, 'W': {}} if is_error_calculated else {}),
                                          {f'{repo.folder/model}/gsa/{kind_name}': {'model': model, 'kind': kind_name}
                                           for kind_name in kind_names for model in models},
                                          ignore_exceptions).over_folders((repo.folder/'gsa'), True)

