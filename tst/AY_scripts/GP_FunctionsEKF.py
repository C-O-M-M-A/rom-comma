import time
import numpy as np
import pandas as pd
from shutil import rmtree, copytree
from pathlib import Path
import romcomma.model as model
from romcomma.data import Store, Fold, Frame
from romcomma.typing_ import NP, Union, Tuple, Sequence


def store_and_fold(source: str, store_name: str, k: int, replace_empty_test_with_data_: bool = True, is_split: bool = True,
                   origin_csv_parameters=None) -> Store:
    """
    Args:
        source: The source csv file name to be used as training data for the GP.
        store_name: The name of the folder where the GP will be stored.
        k: The amount of folds to be used in cross-validation.
        replace_empty_test_with_data_: Whether to replace an empty test file with the training data when K==1.
        is_split: Whether the store needs splitting for multiple outputs.
        origin_csv_parameters: A dictionary stating the index column.
    Returns:
        A ``store'' object which contains a data_csv file, a meta_json file and a standard_csv file. The files contain the global dataset which have
        been split into folds and will be analysed.
    """
    if origin_csv_parameters is None:
        origin_csv_parameters = {'index_col': 0}
    store = Store.from_csv(BASE_PATH / store_name, BASE_PATH / source, **origin_csv_parameters)
    store.standardize(standard=Store.Standard.mean_and_std)
    Fold.into_K_folds(store, k, shuffled_before_folding=True, standard=Store.Standard.none,
                      replace_empty_test_with_data_=replace_empty_test_with_data_)
    if is_split is True:
        store.split()
    return store


def run_gp(store: Store, gp_name: str = "ARD", model_name: str = "ARD", optimize: bool = True, test: bool = True,
           sobol: bool = True, is_split: bool = True, kerneltypeidentifier: str = "gpy_.ExponentialQuadratic"):
    """
    Args:
        store:
        gp_name: The GP name.
        model_name: The name of the Model where the results are being collected.
        optimize:
        test:
        sobol:
        is_split:
        kerneltypeidentifier:
    Returns:
    """

    params = model.base.GP.Parameters(kernel=kernel_parameters, e_floor=-1.0E-3, f=10, e=0.1, log_likelihood=None)
    model.run.GPs(module=model.run.Module.GPY_, name=gp_name, store=store, M_Used=-1, parameters=params, optimize=optimize,
                  test=test, sobol=sobol, optimizer_options=model.gpy_.GP.DEFAULT_OPTIMIZER_OPTIONS)
    model.run.collect_GPs(store=store, model_name=model_name, test=test, sobol=sobol, is_split=is_split, kernelTypeIdentifier=kerneltypeidentifier)
    return


def run_rom(store: Store, rom_name: str = "ROM", gp_name: str = "ARD", it: int = 4,
            guess_it: int = -1, n_exp: int = 3, is_split: bool = False, kerneltypeidentifier: str = "gpy_.ExponentialQuadratic"):
    """
    Args:
        N_exp:
        store: The Store containing the global dataset to be analyzed.
        rom_name: The name of the model where the results are being collected.
        gp_name:
        it: The number of ROM iterations. Each ROM iteration essentially calls Sobol.optimimize(options['sobol_optimizer_options'])
            followed by GP.optimize(options['gp_optimizer_options'])).
        guess_it: After this many ROM iterations, Sobol.optimize does no exploration,
            just gradient descending from Theta = Identity Matrix.
        n_exp:
        is_split: True or False, whether splits have been used in the model.
        kerneltypeidentifier:

    Returns:

            **N_exploit** -- The number of exploratory xi vectors to exploit (gradient descend) in search of the global optimum.
            If N_exploit < 1, only re-ordering of the input basis is allowed.

            **N_explore** -- The number of random_sgn xi vectors to explore in search of the global optimum.
            If N_explore <= 1, gradient descent is initialized from Theta = Identity Matrix.
    """
    sobol_options = {'semi_norm': model.base.Sobol.SemiNorm.DEFAULT_META, 'N_exploit': n_exp, 'N_explore': 100,
                     'options': {'gtol': 1.0E-20}}
    # change N_exploit to below 1 for re-ordering of input basis only instead of rotation
    rom_options = dict(iterations=it, guess_identity_after_iteration=guess_it, sobol_optimizer_options=sobol_options,
                       gp_initializer=model.base.ROM.GP_Initializer.CURRENT_WITH_GUESSED_LENGTHSCALE,
                       gp_optimizer_options=model.run.Module.GPY_.value.GP.DEFAULT_OPTIMIZER_OPTIONS)
    model.run.ROMs(module=model.run.Module.GPY_, name=rom_name, store=store, source_gp_name=gp_name, Mu=-1, Mx=-1,
                   optimizer_options=rom_options)
    model.run.collect_GPs(store=store, model_name=rom_name + ".optimized", test=True, sobol=True, is_split=is_split,
                          kernelTypeIdentifier=kerneltypeidentifier)
    return


if __name__ == '__main__':
    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data")
    STORE_NAME = "x1_z1"
    STORE = store_and_fold("x1_z1.csv", STORE_NAME, 10, is_split=False)
    GP_kernel_name = "ExponentialKroneckerARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), 0.2, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadraticARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), 0.2, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadraticARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), 0.2, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadraticRBF":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), 0.2, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKroneckerARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), 0.2, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "ExponentialKronecker":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, 1), 0.2, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=True, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    """GP should be all done to use either kernel. Just need to try it a few more times -- like without ARD -- then below ROM"""
    # run_rom(store=STORE, rom_name="Reordered_RBF_ROM", gp_name=GP_kernel_name, it=4, guess_it=4, n_exp=0, is_split=False)
    # run_rom(store=STORE, rom_name="Full_RBF_ROM", gp_name=GP_kernel_name, it=4, guess_it=4, n_exp=5, is_split=False)

# if __name__ == '__main__':
#     import numpy as np
#     X = np.array([[1,-0.5],[1,2],[-0.5,1]])
#     S = 0
#     # now for each column vector
#     for i in range(X.shape[1]):
#         x = X[:, i].reshape(-1, 1)
#         xT = x.reshape(1, -1)
#         dif = np.subtract(x, xT)
#         s = (dif != 0).astype(int)
#         S += s