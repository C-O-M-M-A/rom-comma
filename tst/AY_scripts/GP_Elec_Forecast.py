import time
import numpy as np
import pandas as pd
from shutil import rmtree, copytree
from pathlib import Path
import romcomma.model as model
from romcomma.data import Store, Fold, Frame
from romcomma.typing_ import NP, Union, Tuple, Sequence
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster, linkage


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
    Fold.into_K_folds(store, k, shuffled_before_folding=False, standard=Store.Standard.none,
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

    params = model.base.GP.Parameters(kernel=kernel_parameters, e_floor=-1.0E-2, f=1, e=0.01, log_likelihood=None)
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


def k_cluster_step(base_path: Union[str, Path], source: str = "UnS_Train_Data.csv", stand_source: str = "Train_Data.csv",
                   clustered_source: str = "Clustered_Data.csv"):
    """

    Args:
        base_path: The pathway to the file store.
        clustered_source: The name of the csv that the new clustered data will be saved as and will be the source of the following GP.
        source: The initial data used for clustering.
        stand_source: The initial data that has already had it's continuous variables standardised.
        cutoff: A string telling the function which cutoff method to use.
    """
    base_path = Path(base_path)
    df = pd.read_csv(base_path / source, header=[0, 1], index_col=0)  # read the source data file and state the header rows and the index col.
    X = df.copy()  # copy the source file so that the tidying of data that needs clustering doesn't affect the original file.
    X.drop(X.columns[[-1]], axis=1, inplace=True)  # drop the output as that doesn't need clustering
    for i in range(4):
        X.drop(X.columns[[-1]], axis=1, inplace=True)  # drop the date cols

    def standardise(dataset):
        dataNorm = ((dataset - dataset.mean()) / dataset.std())
        # dataNorm[5] = dataset[5] # this may eventually be implemented so that some columns are not standardised.
        return dataNorm

    X = standardise(X)  # standardise the input values using the above function
    X = np.array(X)  # ensure inputs to be clustered are now a ndarray, not a dataframe.
    k = 3
    kmc = KMeans(n_clusters=k, init='random', n_init=5, max_iter=300, tol=1e-04, random_state=0)
    kmc.fit_predict(X)
    membership = kmc.labels_
    df2 = pd.read_csv(base_path / stand_source, header=[0, 1], index_col=0)
    loc = len(df2.columns) - 1
    df2.insert(loc, "X", membership+1)
    df2 = df2.rename({'X': 'Input', '': 'Cluster_No'}, axis=1)
    # now need to save that df as a csv which can be used for GP
    frame = Frame(base_path / clustered_source, df2)
    return frame


def hier_cluster_step(base_path: Union[str, Path], source: str = "UnS_Train_Data.csv", stand_source: str = "Train_Data.csv",
                      clustered_source: str = "Clustered_Data.csv"):
    """

    Args:
        base_path: The pathway to the file store.
        clustered_source: The name of the csv that the new clustered data will be saved as and will be the source of the following GP.
        source: The initial data used for clustering.
        stand_source: The initial data that has already had it's continuous variables standardised.
        cutoff: A string telling the function which cutoff method to use.
    """
    base_path = Path(base_path)
    df = pd.read_csv(base_path / source, header=[0, 1], index_col=0)  # read the source data file and state the header rows and the index col.
    X = df.copy()  # copy the source file so that the tidying of data that needs clustering doesn't affect the original file.
    X.drop(X.columns[[-1]], axis=1, inplace=True)  # drop the output as that doesn't need clustering
    for i in range(4):
        X.drop(X.columns[[-1]], axis=1, inplace=True)  # drop the date cols

    def standardise(dataset):
        dataNorm = ((dataset - dataset.mean()) / dataset.std())
        # dataNorm[5] = dataset[5] # this may eventually be implemented so that some columns are not standardised.
        return dataNorm

    X = standardise(X)  # standardise the input values using the above function
    X = np.array(X)  # ensure inputs to be clustered are now a ndarray, not a dataframe.
    Z = linkage(X, 'ward')
    last = Z[-100:, 2]  # Sub array of last 100 merges
    last_rev = last[::-1]  # Reverse the series
    idxs = np.arange(1, len(last) + 1)
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]  # Reverse the series
    max_acc = max(acceleration_rev)
    n = list(acceleration_rev).index(max_acc) + 2  # + 2 as 0th entry in acceleration_rev is merge 2 -> 1 clusters
    cutoff = (last_rev[n - 1] + last_rev[n - 2]) / 2  # 0th entry in 'last_rev' is 1 cluster containing all vectors
    cluster_membership = fcluster(Z, cutoff, criterion='distance')
    df2 = pd.read_csv(base_path / stand_source, header=[0, 1], index_col=0)
    loc = len(df2.columns) - 1
    df2.insert(loc, "X", cluster_membership)
    df2 = df2.rename({'X': 'Input', '': 'Cluster_No'}, axis=1)
    # now need to save that df as a csv which can be used for GP
    frame = Frame(base_path / clustered_source, df2)
    return frame


if __name__ == '__main__':
    General_Store_Name = "GP-K_Clustering-5folds"
    len_int = 0.2
    Source = 'UnS_Train_Data.csv'
    Stand_Source = "UnS_Train_Data.csv"
    Clustered_Source = 'Clustered_Data-k2.csv'
    """
    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\12-Dec")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("12 -- It took {:.2f} minutes to execute this code.".format(time_mins))
    # GP should be all done to use either kernel. Just need to try it a few more times -- like without ARD -- then below ROM

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\11-Nov")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("12 -- It took {:.2f} minutes to execute this code.".format(time_mins))

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\10-Oct")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("12 -- It took {:.2f} minutes to execute this code.".format(time_mins))

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\9-Sep")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("12 -- It took {:.2f} minutes to execute this code.".format(time_mins))

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\8-Aug")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("12 -- It took {:.2f} minutes to execute this code.".format(time_mins))

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\7-Jul")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("12 -- It took {:.2f} minutes to execute this code.".format(time_mins))

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\6-Jun")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("12 -- It took {:.2f} minutes to execute this code.".format(time_mins))

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\5-May")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("12 -- It took {:.2f} minutes to execute this code.".format(time_mins))
    
    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\4-Apr")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("4 -- It took {:.2f} minutes to execute this code.".format(time_mins))

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\3-Mar")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("3 -- It took {:.2f} minutes to execute this code.".format(time_mins))

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\2-Feb")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = hier_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 1, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=False, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("2 -- It took {:.2f} minutes to execute this code.".format(time_mins))"""

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\1-Jan")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    clustering = k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = General_Store_Name
    STORE = store_and_fold(Clustered_Source, STORE_NAME, 5, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    GP_kernel_name = "ExponentialQuadratic-ARD"
    if GP_kernel_name == "ExponentialQuadratic":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "ExponentialQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialQuadratic"
    elif GP_kernel_name == "RationalQuadratic-ARD":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "RationalQuadratic":
        kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
        kernel_type_identifier = "gpy_.RationalQuadratic"
    elif GP_kernel_name == "ExponentialKronecker-ARD":
        kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.ExponentialKronecker"
    elif GP_kernel_name == "RBFAndEKF-ARD":
        kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
        kernel_type_identifier = "gpy_.RBFAndEKF"
    else:
        kernel_type_identifier = None
        print("This Kernel has not been implemented into the code yet")
    start_GP_time = time.time()
    run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=True, is_split=False,
           kerneltypeidentifier=kernel_type_identifier)
    time_mins = (time.time() - start_GP_time) / 60
    print("1 -- It took {:.2f} minutes to execute this code.".format(time_mins))

    # run_rom(store=STORE, rom_name="Reordered_RBF_ROM", gp_name=GP_kernel_name, it=4, guess_it=4, n_exp=0, is_split=False)
    # run_rom(store=STORE, rom_name="Full_RBF_ROM", gp_name=GP_kernel_name, it=4, guess_it=4, n_exp=5, is_split=False)
"""
BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
STORE_NAME = "Data"
STORE = store_and_fold("Data.csv", STORE_NAME, 5, is_split=True)
# STORE = Store(BASE_PATH / STORE_NAME)
GP_kernel_name = "RBFAndEKF-ARD"
if GP_kernel_name == "ExponentialQuadratic":
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float))
    kernel_type_identifier = "gpy_.ExponentialQuadratic"
elif GP_kernel_name == "ExponentialQuadratic-ARD":
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
    kernel_type_identifier = "gpy_.ExponentialQuadratic"
elif GP_kernel_name == "RationalQuadratic-ARD":
    kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float), power=5)
    kernel_type_identifier = "gpy_.RationalQuadratic"
elif GP_kernel_name == "RationalQuadratic":
    kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=np.full((1, 1), len_int, dtype=float), power=5)
    kernel_type_identifier = "gpy_.RationalQuadratic"
elif GP_kernel_name == "ExponentialKronecker-ARD":
    kernel_parameters = model.gpy_.Kernel.ExponentialKronecker.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
    kernel_type_identifier = "gpy_.ExponentialKronecker"
elif GP_kernel_name == "RBFAndEKF-ARD":
    kernel_parameters = model.gpy_.Kernel.RBFAndEKF.Parameters(lengthscale=np.full((1, STORE.M), len_int, dtype=float))
    kernel_type_identifier = "gpy_.RBFAndEKF"
else:
    kernel_type_identifier = None
    print("This Kernel has not been implemented into the code yet")
start_GP_time = time.time()
run_gp(store=STORE, gp_name=GP_kernel_name, model_name=GP_kernel_name, optimize=True, test=True, sobol=True, is_split=True,
       kerneltypeidentifier=kernel_type_identifier)
       """
