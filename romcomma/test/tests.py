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
from romcomma.test.sampling import latin_hypercube

# BASE_FOLDER = Path('C:\\Users\\fc1ram\\Documents\\Rom\\dat\\SoftwareTest\\DependentGPR')
BASE_FOLDER = Path('C:\\Users\\fc1ram\\Documents\\Rom\\dat\\SiPM')


"""
if __name__ == '__main__':
    with run.Context('Test', float='float64', device='CPU'):
        aggregators = []
        for N in (1000,):
            for M in (5,):
                for noise_magnitude in (0.5,):
                    for is_rotated in (False, ):
                        for broadcast_fraction in ('0.0', '0.1', '0.2', '0.3', '0.4', '0.45', '0.49'):
                            with run.Timing(f'N={N}, noise={noise_magnitude}'):
                                repo = data.storage.Repository(repo_folder(BASE_FOLDER / broadcast_fraction, ['sin.1', 'sin.1', 'sin.2'],
                                                                           N, M, noise_magnitude, is_noise_diagonal=False))
                                run.gsa('initial', repo, False)
"""

if __name__ == '__main__':
    values = latin_hypercube(100, 3)
    result = pd.DataFrame(values)
    result.to_csv(BASE_FOLDER / 'latin_hypercube.csv')

"""
if __name__ == '__main__':
        with run.Context('Test', float='float64'):
        aggregators = []
        csvs = ['S.csv', 'T.csv', 'V.csv', 'Wmm.csv']
        child_path = Path('initial.d.a\\gsa\\first_order.p')
        for N in (1000,):
            for M in (5,):
                for noise_magnitude in (0.5,):
                    for is_rotated in (False, ):
                        for broadcast_fraction in ('0.0', '0.1', '0.2', '0.3', '0.4', '0.45', '0.49'):
                            with run.Timing(f'N={N}, noise={noise_magnitude}'):
                                repo = repo_folder(BASE_FOLDER / broadcast_fraction, ['sin.1', 'sin.1', 'sin.2'], N, M, noise_magnitude, is_noise_diagonal=False)
                                data.storage.Repository(repo).aggregate_over_folds(child_path, csvs, is_K_included=True, header=0)
                                aggregators.append({'folder': repo / child_path, 'broadcast_fraction': float(broadcast_fraction)})

        run.aggregate(dict.fromkeys(csvs, aggregators),
                      BASE_FOLDER / Path('initial.d.a\\gsa\\first_order.p'), header=0)
                                # repo = Repository(repo_folder(BASE_FOLDER, ['sin.1', 'sin.1'], N, M, noise_magnitude, is_noise_diagonal=False))
                                # run.gpr2(name='initial', repo=repo, is_read=None, is_isotropic=None, is_independent=False,
                                #          broadcast_fraction=float(broadcast_fraction), optimize=False, test=True)
                                # run.gsa('sin', repo, is_independent=True)
                                # run.gsa('sin', repo, is_independent=False)

"""
"""
if __name__ == '__main__':
    with run.Context('Test', float='float64'):
        for N in (1000,):
            for M in (5,):
                for noise_magnitude in (0.5,):
                    for is_rotated in (False, ):
                        for broadcast_fraction in ('0.0', '0.1', '0.2', '0.3', '0.4', '0.45', '0.49'):
                            with run.Timing(f'N={N}, noise={noise_magnitude}'):
                                repo = data.storage.Repository(repo_folder(BASE_FOLDER / broadcast_fraction, ['sin.1', 'sin.1', 'sin.2'], N, M,
                                              noise_magnitude=noise_magnitude, is_noise_diagonal=False, is_noise_variance_stochastic=False))
                                repo.aggregate_over_folds('initial.d.a', ['test_summary.csv'], is_K_included=True, header=[0, 1])
                                # repo = Repository(repo_folder(BASE_FOLDER, ['sin.1', 'sin.1'], N, M, noise_magnitude, is_noise_diagonal=False))
                                # run.gpr2(name='initial', repo=repo, is_read=None, is_isotropic=None, is_independent=False,
                                #          broadcast_fraction=float(broadcast_fraction), optimize=False, test=True)
                                # run.gsa('sin', repo, is_independent=True)
                                # run.gsa('sin', repo, is_independent=False)
"""
"""
if __name__ == '__main__':
    with run.Context('Test', float='float64'):
        for N in (1000,):
            for M in (5,):
                for noise_magnitude in (0.5,):
                    for is_rotated in (False, ):
                        for broadcast_fraction in ('0.0', '0.1', '0.2', '0.4', '0.6', '0.8', '0.99'):
                            with run.Timing(f'N={N}, noise={noise_magnitude}'):
                                repo = sample(BASE_FOLDER, ['sin.1', 'sin.1', 'sin.2'], N, M, K=2,
                                              noise_magnitude=noise_magnitude, is_noise_diagonal=False, is_noise_variance_stochastic=False)
                                # repo = Repository(repo_folder(BASE_FOLDER, ['sin.1', 'sin.1'], N, M, noise_magnitude, is_noise_diagonal=False))
                                run.gpr(name='initial', repo=repo, is_read=None, is_isotropic=None, is_independent=True, optimize=True, test=True)
                                # run.gsa('sin', repo, is_independent=True)
                                # run.gsa('sin', repo, is_independent=False)
"""