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

""" Contains developer tests of romcomma. """
import pandas as pd

from romcomma.base.definitions import *
from romcomma import run, data
from romcomma.test.utilities import repo_folder
from romcomma.test.utilities import sample

BASE_FOLDER = Path('C:\\Users\\fc1ram\\Documents\\Rom\\dat\\SoftwareTest\\Dependency\\1.1')


if __name__ == '__main__':
    with run.Context('Test', device='CPU'):
        kinds = [run.perform.GSA.Kind.FIRST_ORDER, run.perform.GSA.Kind.CLOSED, run.perform.GSA.Kind.TOTAL]
        kind_names = [kind.name.lower() for kind in kinds]
        models = ['diag.i.i', 'diag.i.a', 'diag.d.a', 'variance.d.a', 'lengthscales.d.a']
        for N in (400,):
            for M in (5,):
                for noise_magnitude in (0.1,):
                    for is_noise_diagonal in (True, False):
                        with run.Timing(f'N={N}, noise={noise_magnitude} \n'):
                            # repo = sample(BASE_FOLDER, ['s.0', 's.1'], N, M, K=-2,
                            #               noise_magnitude=noise_magnitude, is_noise_diagonal=is_noise_diagonal, is_noise_variance_stochastic=True)
                            repo = data.storage.Repository(repo_folder(BASE_FOLDER, ['s.0', 's.1'], N, M,
                                        noise_magnitude=noise_magnitude, is_noise_diagonal=is_noise_diagonal, is_noise_variance_stochastic=True))

                            # run.gpr(name='diag', repo=repo, is_read=None, is_independent=None, is_isotropic=None, optimize=True, test=True)
                            # run.copy('diag.d.a', 'variance.d.a', repo)
                            # run.gpr(name='variance', repo=repo, is_read=True, is_independent=False, is_isotropic=False, optimize=True, test=True,
                            #         kernel={'variance': {'diagonal': True, 'off_diagonal': True}})
                            # run.copy('variance.d.a', 'lengthscales.d.a', repo)
                            # run.gpr(name='lengthscales', repo=repo, is_read=True, is_independent=False, is_isotropic=False, optimize=True, test=True,
                            #         kernel={'lengthscales': {'independent': True, 'dependent': True}})
                            #
                            # aggregators= {'test_summary.csv' : [{'folder': repo.folder / model, 'model': model, 'kwargs': {'header': [0, 1], 'index_col': 0}}
                            #                                    for model in models]}
                            # run.aggregate(aggregators=aggregators, dst=repo.folder / 'gpr', ignore_missing=False)
                            # aggregators = {'variance.csv': None, 'log_marginal.csv': None}
                            # for key in aggregators.keys():
                            #     aggregators[key] = [{'folder': (repo.folder / model) / 'likelihood', 'model': model} for model in models]
                            # run.aggregate(aggregators=aggregators, dst=(repo.folder / 'gpr') / 'likelihood', ignore_missing=False)
                            #
                            # aggregators = {'variance.csv': None, 'lengthscales.csv': None}
                            # for key in aggregators.keys():
                            #     aggregators[key] = [{'folder': (repo.folder / model) / 'kernel', 'model': model} for model in models]
                            # run.aggregate(aggregators=aggregators, dst=(repo.folder / 'gpr') / 'kernel', ignore_missing=False)
                            #
                            # run.gsa('diag', repo, is_independent=None, is_isotropic=None, kinds=kinds)
                            run.gsa('variance', repo, is_independent=False, is_isotropic=False, kinds=kinds, is_F_diagonal=False)
                            # run.gsa('lengthscales', repo, is_independent=False, is_isotropic=False, kinds=kinds, is_F_diagonal=True)
                            # aggregators = {}
                            # for key in ['S.csv', 'V.csv']:
                            #     aggregators[key] = [{'folder': (((repo.folder / name) / 'gsa') / kind_name), 'model': name, 'kind': kind_name}
                            #                         for kind_name in kind_names
                            #                         for name in ['diag.i.i', 'diag.i.a', 'diag.d.a', 'variance.d.a', 'lengthscales.d.a']]
                            # run.aggregate(aggregators=aggregators, dst=repo.folder / 'gsa')


# if __name__ == '__main__':
#     with run.Context('Test', device='CPU'):
#         kinds = [run.perform.GSA.Kind.FIRST_ORDER, run.perform.GSA.Kind.CLOSED, run.perform.GSA.Kind.TOTAL]
#         kind_names = [kind.name.lower() for kind in kinds]
#         models = ['diag.i.i', 'diag.i.a', 'diag.d.a', 'variance.d.a', 'lengthscales.d.a']
#         for N in (400,):
#             for M in (5,):
#                 for noise_magnitude in (0.1,):
#                     for is_noise_diagonal in (True, False):
#                         with run.Timing(f'N={N}, noise={noise_magnitude} \n'):
#                             # repo = sample(BASE_FOLDER, ['s.0', 's.1'], N, M, K=-2,
#                             #               noise_magnitude=noise_magnitude, is_noise_diagonal=is_noise_diagonal, is_noise_variance_stochastic=True)
#                             repo = data.storage.Repository(repo_folder(BASE_FOLDER, ['s.0', 's.1'], N, M,
#                                           noise_magnitude=noise_magnitude, is_noise_diagonal=is_noise_diagonal, is_noise_variance_stochastic=True))
#                             run.gsa('diag', repo, is_independent=None, is_isotropic=None, kinds=kinds)
#                             run.gsa('variance', repo, is_independent=False, is_isotropic=False, kinds=kinds)
#                             run.gsa('lengthscales', repo, is_independent=False, is_isotropic=False, kinds=kinds)
#                             aggregators = {}
#                             for key in ['S.csv', 'V.csv']:
#                                 aggregators[key] = [{'folder': (((repo.folder / name) / 'gsa') / kind_name), 'model': name, 'kind': kind_name}
#                                                     for kind_name in kind_names
#                                                     for name in ['diag.i.i', 'diag.i.a', 'diag.d.a', 'variance.d.a', 'lengthscales.d.a']]
#                             run.aggregate(aggregators=aggregators, dst=repo.folder / 'gsa')
