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

BASE_FOLDER = Path('C:/Users/fc1ram/Documents/Rom/dat/Jude EV')

# if __name__ == '__main__':
#     with run.Context('Test'):
#         kinds = [run.perform.GSA.Kind.FIRST_ORDER, run.perform.GSA.Kind.CLOSED, run.perform.GSA.Kind.TOTAL]
#         kind_names = [kind.name.lower() for kind in kinds]
#         models = ['2WP.0', '2WP.1']
#         name = 'model'
#         for model in models:
#             with run.TimingOneLiner(f'{model} \n'):
#                 repo = data.storage.Repository.from_csv(BASE_FOLDER / model, BASE_FOLDER / f'{model}.csv')
#                 repo.into_K_folds(K=2, shuffle_before_folding=True)
#                 run.gpr(name=name, repo=repo, is_read=None, is_independent=True, is_isotropic=None, optimize=True, test=True)
#                 # repo = data.storage.Repository(BASE_FOLDER / model)
#                 # run.gsa('model', repo, is_independent=True, is_isotropic=False, kinds=kinds)
#
#         aggregators = {'test_summary.csv' : [{'folder': (BASE_FOLDER / model) / f'{name}.i.a', 'model': model, 'kwargs': {'header': [0, 1], 'index_col': 0}}
#                                            for model in models]}
#         run.aggregate(aggregators=aggregators, dst=BASE_FOLDER / 'gpr', ignore_missing=False)
#         aggregators = {'variance.csv': None, 'log_marginal.csv': None}
#         for key in aggregators.keys():
#             aggregators[key] = [{'folder': ((BASE_FOLDER / model) / f'{name}.i.a') / 'likelihood', 'model': model} for model in models]
#         run.aggregate(aggregators=aggregators, dst=(BASE_FOLDER / 'gpr') / 'likelihood', ignore_missing=False)
#
#         aggregators = {'variance.csv': None, 'lengthscales.csv': None}
#         for key in aggregators.keys():
#             aggregators[key] = [{'folder': ((BASE_FOLDER / model) / f'{name}.i.a') / 'kernel', 'model': model} for model in models]
#         run.aggregate(aggregators=aggregators, dst=(BASE_FOLDER / 'gpr') / 'kernel', ignore_missing=False)
#
#         # aggregators = {}
#         # for key in ['S.csv', 'V.csv']:
#         #     aggregators[key] = [{'folder': (((BASE_FOLDER / model) / f'{name}.i.a') / 'gsa') / kind_name, 'model': model, 'kind': kind_name}
#         #                         for kind_name in kind_names
#         #                         for model in models]
#         #     run.aggregate(aggregators=aggregators, dst=BASE_FOLDER / 'gsa')

if __name__ == '__main__':
    with run.Context('Test', device='CPU'):
        kinds = [run.perform.GSA.Kind.FIRST_ORDER, run.perform.GSA.Kind.CLOSED, run.perform.GSA.Kind.TOTAL]
        kind_names = [kind.name.lower() for kind in kinds]
        models = ['2WP.0', '2WP.1']
        name = 'model'
        for model in models:
            with run.TimingOneLiner(f'{model} \n'):
                # repo = data.storage.Repository.from_csv(BASE_FOLDER / model, BASE_FOLDER / f'{model}.csv')
                # repo.into_K_folds(K=2, shuffle_before_folding=True)
                # run.gpr(name=name, repo=repo, is_read=None, is_independent=True, is_isotropic=None, optimize=True, test=True)
                repo = data.storage.Repository(BASE_FOLDER / model)
                run.gsa('model', repo, is_independent=True, is_isotropic=False, kinds=kinds)

        # aggregators = {'test_summary.csv' : [{'folder': (BASE_FOLDER / model) / f'{name}.i.a', 'model': model, 'kwargs': {'header': [0, 1], 'index_col': 0}}
        #                                    for model in models]}
        # run.aggregate(aggregators=aggregators, dst=BASE_FOLDER / 'gpr', ignore_missing=False)
        # aggregators = {'variance.csv': None, 'log_marginal.csv': None}
        # for key in aggregators.keys():
        #     aggregators[key] = [{'folder': ((BASE_FOLDER / model) / f'{name}.i.a') / 'likelihood', 'model': model} for model in models]
        # run.aggregate(aggregators=aggregators, dst=(BASE_FOLDER / 'gpr') / 'likelihood', ignore_missing=False)
        #
        # aggregators = {'variance.csv': None, 'lengthscales.csv': None}
        # for key in aggregators.keys():
        #     aggregators[key] = [{'folder': ((BASE_FOLDER / model) / f'{name}.i.a') / 'kernel', 'model': model} for model in models]
        # run.aggregate(aggregators=aggregators, dst=(BASE_FOLDER / 'gpr') / 'kernel', ignore_missing=False)

        aggregators = {}
        for key in ['S.csv', 'V.csv']:
            aggregators[key] = [{'folder': (((BASE_FOLDER / model) / f'{name}.i.a') / 'gsa') / kind_name, 'model': model, 'kind': kind_name}
                                for kind_name in kind_names
                                for model in models]
            run.aggregate(aggregators=aggregators, dst=BASE_FOLDER / 'gsa')

