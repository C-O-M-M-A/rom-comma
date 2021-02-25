
"""This script:
 1: Imports n lists of 24 x d hourly data points
 2: For each list, normalised based on maximum element
 3: For each d, take points 24d - to 24(d+1) for each n and form list of length 24 x n
 4: Make list of lists of length 24 x n to get 24 x n X d array
 5: perform cluster analysis on this array"""
#########################
# Import required tools #
#########################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, is_monotonic    # For cluster analysis
from scipy.cluster.hierarchy import fcluster   # For cluster membership
from matplotlib import pyplot as plt
import tkinter as _tkinter
import math
from pathlib import Path
import time
from numpy import full
import romcomma.model as model
from romcomma.data import Store, Fold, Frame
from romcomma.typing_ import NP, Union, Tuple, Sequence
from shutil import rmtree, copytree
from scipy import optimize
from pyDOE import lhs
from scipy.stats.distributions import norm
from sklearn.metrics import r2_score



def pre_cluster_step(BASE_PATH: Union[str, Path], SOURCE: str, CLUSTERED_SOURCE: str = "Clustered_Data.csv", cutoff='acceleration'):
    """

    Args:
        CLUSTERED_SOURCE:
        SOURCE:
        BASE_PATH:
        cutoff:

    Returns:

    """
    BASE_PATH = Path(BASE_PATH)
    df = pd.read_csv(BASE_PATH / SOURCE, header=[0, 1], index_col=0)  # read the source data file and state the header rows and the index col.
    X = df.copy() # copy the source file so that the tidying of data that needs clustering doesn't affect the original file.
    X.drop(X.columns[[-1]], axis=1, inplace=True)  # drop the output as that doesn't need clustering

    def standardise(dataset):
        dataNorm = ((dataset - dataset.mean()) / dataset.std())
        # dataNorm[5] = dataset[5] # this may eventually be implemented so that some columns are not standardised.
        return dataNorm

    X = standardise(X)  # standardise the input values using the above function
    input_array = np.array(X)  # ensure inputs to be clustered are now a ndarray, not a dataframe.
    # Generate linkage matrix (this does all work, after this it's just a case of defining cluster cut-off points)
    Z = linkage(input_array, 'ward')
    '''Below we use the fcluster function to get the cluster membership. We must set a cutoff, i.e. a threshold for 
    judging a cluster to be genuinely separate. The fcluster function takes a merge distance as it it's cutoff, i.e. 
    merges with distance >= cutoff have joined two genuine clusters.'''

    # define cutoff point for intercluster distance
    cutoff = 1500
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')

    def fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Sample index')
            plt.ylabel('Distance £/MWh')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata

    fancy_dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        no_labels=True,  # Suppress singleton cluster labels that clog up x -axis
        annotate_above=cutoff,  # Prevent excessive annotation of merge distances
        max_d=cutoff)
    plt.show(block=False)
    # Elbow plot, for guidance on where to truncate dendrogram

    last = Z[-100:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)

    plt.figure(figsize=(25, 10))
    plt.title('Elbow plot')
    plt.xlabel('Clusters identified')
    plt.ylabel('Distance travelled to join clusters')
    plt.plot(idxs, last_rev, label="Dist", marker='D')

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev, label="2nd Deriv Dist", marker='.')
    plt.legend()
    plt.show(block=False)

    print("idxs", idxs)
    print("last_rev", last_rev)
    print("accel_rev", acceleration_rev)
    """
    last = Z[-100:, 2] # Sub array of last 100 merges
    last_rev = last[::-1] # Reverse the series
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1] # Reverse the series
    """
    if cutoff=='manual':
        print('manual')
        n=int(input('Select number of clusters (i.e. last n-1 merges)'))
        # Select merge distance corresponding to merge that takes us from n clusters to n-1
        cutoff = (last_rev[n-1] + last_rev[n-2])/2 # 0th entry in 'last_rev' is 1 cluster containing all vectors
    elif cutoff == 'acceleration':
        # Pullout index of maximum acceleration
        max_acc = max(acceleration_rev)
        n = list(acceleration_rev).index(max_acc) + 2  # + 2 as 0th entry in acceleration_rev is merge 2 -> 1 clusters
        cutoff = (last_rev[n-1] + last_rev[n-2])/2  # 0th entry in 'last_rev' is 1 cluster containing all vectors
    # Retrieve cluster membership for each vector (day in our example)
    cluster_membership = fcluster(Z, cutoff, criterion='distance')
    # retrieve the number of clusters
    cluster_list = []
    for i in cluster_membership:
        if i not in cluster_list:
            cluster_list += [i]
    n = len(cluster_list)
    print("number of clusters = ", n)
    df.insert(0, "X", cluster_membership)
    df = df.rename({'X': 'Input', '': 'Cluster_No'}, axis=1)
    # now need to save that df as a csv which can be used for GP
    frame = Frame(BASE_PATH / CLUSTERED_SOURCE, df)
    return frame


def store_and_fold(BASE_PATH: Union[str, Path], SOURCE: str, STORE_NAME: str, K: int, replace_empty_test_with_data_: bool = True, is_split: bool = True,
                   ORIGIN_CSV_PARAMETERS=None) -> Store:
    """
    Args:
        BASE_PATH:
        SOURCE: The source csv file name to be used as training data for the GP.
        STORE_NAME: The name of the folder where the GP will be stored.
        K: The amount of folds to be used in cross-validation.
        replace_empty_test_with_data_: Whether to replace an empty test file with the training data when K==1.
        is_split: Whether the store needs splitting for multiple outputs.
        ORIGIN_CSV_PARAMETERS: A dictionary stating the index column.
    Returns:
        A ``store'' object which contains a data_csv file, a meta_json file and a standard_csv file. The files contain the global dataset which have
        been split into folds and will be analysed.
    """
    if ORIGIN_CSV_PARAMETERS is None:
        ORIGIN_CSV_PARAMETERS = {'index_col': 0}
    store = Store.from_csv(BASE_PATH / STORE_NAME, BASE_PATH / SOURCE, **ORIGIN_CSV_PARAMETERS)
    # store.standardize(standard=Store.Standard.mean_and_std)
    Fold.into_K_folds(store, K, shuffled_before_folding=False, standard=Store.Standard.none,
                      replace_empty_test_with_data_=replace_empty_test_with_data_)
    if is_split is True:
        store.split()
    return store


def run_GP(STORE: Store, GP_NAME_STEM: str = "ARD", optimize: bool = True, test: bool = True, sobol: bool = True, is_split: bool = True):
    """
    Args:
        STORE:
        GP_NAME_STEM:
        optimize:
        test:
        sobol:
        is_split:
    Returns:

    """
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, STORE.M), 0.2, dtype=float))
    params = model.base.GP.Parameters(kernel=kernel_parameters, e_floor=-1.0E-4, f=1.0, e=0.1, log_likelihood=None)
    model.run.GPs(module=model.run.Module.GPY_, name=GP_NAME_STEM, store=STORE, M_Used=-1, parameters=params, optimize=optimize, test=test,
                  sobol=sobol, optimizer_options=model.gpy_.GP.DEFAULT_OPTIMIZER_OPTIONS)
    if test is True:
        model.run.collect_tests(store=STORE, model_name=GP_NAME_STEM, is_split=is_split)
    model.run.collect(store=STORE, model_name=GP_NAME_STEM, parameters=model.gpy_.GP.DEFAULT_PARAMETERS, is_split=is_split)
    model.run.collect(store=STORE, model_name=Path(GP_NAME_STEM) / model.gpy_.GP.KERNEL_NAME, parameters=kernel_parameters, is_split=is_split)
    model.run.collect(store=STORE, model_name=Path(GP_NAME_STEM) / model.base.Sobol.NAME, parameters=model.base.Sobol.DEFAULT_PARAMETERS,
                      is_split=is_split)
    return


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


def Test_Data(gp_path: Union[str, Path], test_data_file: str, standardized: bool, standard_data_file: str = "__standard__.csv"):
    """ Loads and saves the test data as standardized inputs (X) and standardized observed outputs (Y).

    Args:
        standardized: Is the test data already standardised?
        gp_path: Path to the folder (inside a split) where the test data and standard data files are stored.
        test_data_file: The name of the test data file, e.g. 'test_data.csv'.
        standard_data_file: The name of the standard data file which contains the mean and standard deviation used to standardize the test data.
    Returns: A tuple containing the NP.Matrix of inputs (X) and the NP.Vector of outputs (Y).
    """
    GP_PATH = Path(gp_path)
    Split_Store = Store(GP_PATH)
    fold = Fold(Split_Store, k)
    fold_dir = fold.dir
    test_data = pd.read_csv(fold_dir / test_data_file, header=[0, 1], index_col=0)
    if standardized is True:
        Dataset = test_data.values
        Stand_Inputs = Dataset[:, :-1]
        Stand_Observed_Outputs = Dataset[:, -1]
        return Stand_Inputs, Stand_Observed_Outputs
    else:
        mean_std = pd.read_csv(GP_PATH / standard_data_file, index_col=0)
        Dataset = test_data.values
        AVG_STD = mean_std.values
        Stand_Dataset = (Dataset - AVG_STD[1].astype(float)) / AVG_STD[2].astype(float)
        Stand_Inputs = Stand_Dataset[:, :-1]
        Stand_Observed_Outputs = Stand_Dataset[:, -1]
        return Stand_Inputs, Stand_Observed_Outputs


if __name__ == '__main__':
    start = time.time()
    Base_Path = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing\\1-Jan")
    store_name = "GPc1"
    Source = 'UnS_Train_Data.csv'
    Clustered_Source = 'Clustered_Data.csv'
    clustering = pre_cluster_step(Base_Path, Source, cutoff='Acceleration')
    """STORE = store_and_fold(Base_Path, Clustered_Source, store_name, 1, is_split=False)
    start_GP_time = time.time()
    # run_GP(store, is_split=False)
    GP_kernel_name = "ExponentialQuadraticARD"
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
    end_GP_time = (time.time() - start_GP_time) / 60
    print("GP finished training in {:.2f} minutes.".format(end_GP_time))"""
"""
    splits = store.splits
    K = store.meta['K']
    test_name = "__test__.csv"
    csv_name = "Test_Predictions.csv"
    pred_folder = "GP_Tests"
    for split_dir in splits:
        split_start = time.time()
        GP_Path = Path(Base_Path / store_name / split_dir[1])
        for k in range(K):
            #GP_Path = Path(Base_Path)
            fold_start = time.time()
            Mu = -1
            X, Y = Test_Data(gp_path=GP_Path, test_data_file=test_name, standardized=True)
            # X = Test_Data_No_Outputs(gp_path=GP_Path, test_data_file=test_name, standardized=True)
            # X = np.array([-0.15752322, -0.31259307, 1.47047284, -3.5])
            # U = Rotate_Inputs(gp_path=GP_Path, Stand_Inputs=X, Us_taken=Mu)
            Prediction = Predict(gp_path=GP_Path, gb_source="ARD", gb_destination=pred_folder, Rotated_Inputs=X,
                                 Us_taken=Mu, k=k)
            predict_frame = Create_CSVs(gp_path=GP_Path, gb_destination=pred_folder, Predicted_Outputs=Prediction,
                                        Stand_Inputs=X, Rotated_Inputs=X, csv_name=csv_name, k=k)
            # predict_frame = Create_CSV_with_Observed(gp_path=GP_Path, gb_destination=pred_folder, Predicted_Outputs=Prediction,
            #                                         Stand_Inputs=X, Rotated_Inputs=U, Stand_Observed_Outputs=Y, csv_name=csv_name, k=k)
            # print("Split", split_dir[0], "Fold", k, "has a goodness of fit = ", r2_score(Y, Prediction[0]))
            fold_time_mins = (time.time() - fold_start) / 60
            print("Fold", k, "has finished in {:.2f} minutes.".format(fold_time_mins))
            print()
        split_time_mins = (time.time() - split_start) / 60
        print()
        print("split.", split_dir[0], "has finished in {:.2f} minutes.".format(split_time_mins))
        print()
        print()
    collect_predictions(store=store, folder_name=pred_folder, csv_name=csv_name, is_split=True)
    time_mins = (time.time() - start) / 60
    print("It took {:.2f} minutes to execute this code.".format(time_mins))
"""
### should be at a position now where I can use the pre-cluster step straight into the GP. Just needs a lot of tidying.