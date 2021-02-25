# BSD 3-Clause License
#
# Copyright (c) 2019, Robert A. Milton
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Run this module first thing, to test your installation of romcomma.

**Contents**:
    **predict**: Prediction using a GaussianBundle.

    **test_input**: A rudimentary test input, for installation testing.
"""

from pathlib import Path
from numpy import full
from romcomma import data, model

EFFECTIVELY_ZERO = 1.0E-64
BASE_PATH = Path('X:\\comma_group1\\Rom\\dat\\Imperial')


if __name__ == '__main__':
    store_name = BASE_PATH / "3"
    # name = 'rbf'
    # kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, 1), 2.0, dtype=float))
    # parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=1E-7, e=0.01)
    # store = data.Store.from_csv(store_name, BASE_PATH / "202005.2.csv", index_col=0)
    # data.Fold.into_K_folds(parent=store, K=5, shuffled_before_folding=True, standard=data.Store.Standard.none, replace_empty_test_with_data_=True)
    # model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=parameters, optimize=True, test=True, sobol=True,
    #               optimizer_options={'optimizer': 'bfgs', 'max_iters': 5000, 'gtol': 1E-16})
    # model.run.GPs(module=model.run.Module.GPY_, name=name, store=store, M=-1, parameters=None, optimize=True, test=True, sobol=True,
    #               optimizer_options={'optimizer': 'bfgs', 'max_iters': 5000, 'gtol': 1E-16}, make_ard=True)
    # model.run.collect_GPs(store=store, model_name=name, test=True, sobol=True, is_split=False)
    # model.run.collect_GPs(store=store, model_name=name + ".ard", test=True, sobol=True, is_split=False)
    store = data.Store(store_name)
    # sobol_options = {'semi_norm': model.base.Sobol.SemiNorm.DEFAULT_META, 'N_exploit': 0, 'N_explore': 0, 'options': {'gtol': 1.0E-16}}
    # rom_options = {'iterations': 3, 'guess_identity_after_iteration': 4, 'sobol_optimizer_options': sobol_options,
    #                              'gp_initializer': model.base.ROM.GP_Initializer.CURRENT_WITH_GUESSED_LENGTHSCALE,
    #                              'gp_optimizer_options': {'optimizer': 'bfgs', 'max_iters': 5000, 'gtol': 1E-16}}
    # model.run.ROMs(module=model.run.Module.GPY_, name='rom.reorder', store=store, source_gp_name='rbf.ard', Mu=-1, Mx=-1,
    #                optimizer_options=rom_options)
    model.run.collect_GPs(store=store, model_name="rom.reorder.optimized", test=True, sobol=True, is_split=False)
