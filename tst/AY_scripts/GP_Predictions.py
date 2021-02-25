import romcomma.model as model
import numpy as np
import pandas as pd
from romcomma.data import Store, Fold, Frame
from romcomma.typing_ import NP, Union, Tuple, List
from shutil import rmtree, copytree
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster, linkage


def test_data(gp_path: Union[str, Path], test_data_file: str, standardized: bool, k: int, standard_data_file: str = "__standard__.csv"):
    """ Loads and saves the test data as standardized inputs (X) and standardized observed outputs (Y).

    Args:
        k:
        standardized: Is the test data already standardised?
        gp_path: Path to the folder (inside a split) where the test data and standard data files are stored.
        test_data_file: The name of the test data file, e.g. 'test_data.csv'.
        standard_data_file: The name of the standard data file which contains the mean and standard deviation used to standardize the test data.
    Returns: A tuple containing the NP.Matrix of inputs (X) and the NP.Vector of outputs (Y).
    """
    gp_path = Path(gp_path)
    split_store = Store(gp_path)
    fold = Fold(split_store, k)
    fold_dir = fold.dir
    test_data_csv = pd.read_csv(BASE_PATH / test_data_file, header=[0, 1], index_col=0)
    if standardized is True:
        dataset = test_data_csv.values
        stand_inputs = dataset[:, :-1]
        stand_observed_outputs = dataset[:, -1]
        return stand_inputs, stand_observed_outputs
    else:
        mean_std = pd.read_csv(gp_path / standard_data_file, index_col=0)
        dataset = test_data_csv.values
        avg_std = mean_std.values
        stand_dataset = (dataset - avg_std[1].astype(float)) / avg_std[2].astype(float)
        stand_inputs = stand_dataset[:, :-1]
        stand_observed_outputs = stand_dataset[:, -1]
        return stand_inputs, stand_observed_outputs


def test_data_no_outputs(gp_path: Union[str, Path], test_data_file: str, standardized: bool, k: int, standard_data_file: str = "__standard__.csv"):
    """ Loads and saves the test data as standardized inputs (X) and standardized observed outputs (Y).

    Args:
        k:
        standardized: Is the test data already standardised?
        gp_path: Path to the folder (inside a split) where the test data and standard data files are stored.
        test_data_file: The name of the test data file, e.g. 'test_data.csv'.
        standard_data_file: The name of the standard data file which contains the mean and standard deviation used to standardize the test data.
    Returns: A tuple containing the NP.Matrix of inputs (X) and the NP.Vector of outputs (Y).
    """
    gp_path = Path(gp_path)
    split_store = Store(gp_path)
    fold = Fold(split_store, k)
    fold_dir = fold.dir
    input_test_data = pd.read_csv(BASE_PATH / test_data_file, header=[0, 1], index_col=0)
    if standardized is True:
        dataset = input_test_data.values
        stand_inputs = dataset[:, :]
        return stand_inputs
    else:
        mean_std = pd.read_csv(gp_path / standard_data_file, index_col=0)
        dataset = input_test_data.values
        avg_std = mean_std.values
        stand_dataset = (dataset - avg_std[1].astype(float)) / avg_std[2].astype(float)
        stand_inputs = stand_dataset[:, :]
        return stand_inputs


def predict(gp_path: Union[str, Path], gb_source: str, gb_destination: str, rotated_inputs: NP.Matrix, Us_taken: int, k: int = 0)\
            -> Tuple[NP.Matrix, NP.Matrix, NP.Tensor3]:
    """ Prediction using a GaussianBundle.

    Args:
        gp_path: The path that takes you into the trained Gaussian process' Split folder where the Fold can be found.
        gb_source: The name of the trained GB e.g. "ROM.optimized" or "ARD".
        gb_destination: The name of the new folder where the GB will be copied and predictions made e.g. "ROM_Predictions".
        rotated_inputs: The rotated inputs that are predicted on.
        Us_taken: The amount of Us taken from the ROM to be used in predicting.
        k: The fold that contains the data that has been used to train the GP's.
    Returns: A pair (predictive_mean, predictive_std) of (N,1) numpy arrays.

    """
    gp_path = Path(gp_path)
    split_store = Store(gp_path)
    fold = Fold(split_store, k, Us_taken)
    gb_dir = fold.dir / gb_destination
    rmtree(gb_dir, ignore_errors=True)
    copytree(src=fold.dir / gb_source, dst=gb_dir)
    gb = model.gpy_.GP(fold=fold, name=gb_destination, parameters=None)
    predicted_output = gb.predict(rotated_inputs)
    return predicted_output


def create_csv(gp_path: Union[str, Path], gb_destination: str, predicted_outputs: Tuple[NP.Vector, NP.Vector, NP.Tensor3], stand_inputs: NP.Matrix,
               rotated_inputs: NP.Matrix, k: int = 0, csv_name: str = "__predictions__"):
    """ Saves the inputs, rotated inputs and predictions into a CSV file.

    Args:
        gp_path: The path that takes you into the trained Gaussian process' Split folder where the Fold can be found.
        gb_destination: The name of the new folder where the GB will be copied and predictions made e.g. "ROM_Predictions".
        predicted_outputs: A pair (predictive_mean, predictive_std) of (N,1) numpy arrays.
        stand_inputs: The standardized input values - an (N,M) numpy array, consisting of N test inputs, each of dimension M.
        rotated_inputs: The rotated inputs that are predicted on.
        k: The fold that contains the data that has been used to train the GP's.
        csv_name: The name of the csv file that has been created. Default is "__predictions__".
    Returns: A frame containing the inputs, the rotated inputs and predictions for each split directory.

    """
    gp_path = Path(gp_path)
    split_store = Store(gp_path)
    fold = Fold(split_store, k)
    gb_dir = fold.dir / gb_destination
    mean_prediction_T = predicted_outputs[0]
    std_prediction_T = np.sqrt(predicted_outputs[1])
    df_X = pd.DataFrame(stand_inputs)
    inputs_label = ['x' + str(i - 1) for i in np.arange(1, len(df_X.columns) + 1)]
    inputs_dict = dict(zip(df_X.columns, inputs_label))
    df_X = df_X.rename(columns=inputs_dict)
    list_input_header = list(df_X.columns.values)
    for idx, val in enumerate(list_input_header):
        list_input_header[idx] = 'Stand_Input'
    df_X.columns = pd.MultiIndex.from_tuples(zip(list_input_header, df_X.columns))
    """df_U = pd.DataFrame(Rotated_Inputs)
    rotated_label = ['u' + str(i - 1) for i in np.arange(1, len(df_U.columns) + 1)]
    rotated_dict = dict(zip(df_U.columns, rotated_label))
    df_U = df_U.rename(columns=rotated_dict)
    list_input_header = list(df_U.columns.values)
    for idx, val in enumerate(list_input_header):
        list_input_header[idx] = 'Rotated_Input'
    df_U.columns = pd.MultiIndex.from_tuples(zip(list_input_header, df_U.columns))"""
    df_mean = pd.DataFrame(mean_prediction_T)
    mean_label = ['Predicted_Mean']
    mean_dict = dict(zip(df_mean.columns, mean_label))
    df_mean = df_mean.rename(columns=mean_dict)
    df_std = pd.DataFrame(std_prediction_T)
    std_label = ['Predicted_STD']
    std_dict = dict(zip(df_std.columns, std_label))
    df_std = df_std.rename(columns=std_dict)
    df_outputs = pd.concat([df_mean, df_std], axis=1)
    list_output_header = list(df_outputs.columns.values)
    for idx, val in enumerate(list_output_header):
        list_output_header[idx] = 'Output'
    df_outputs.columns = pd.MultiIndex.from_tuples(zip(list_output_header, df_outputs.columns))
    df = pd.concat([df_X, df_outputs], axis=1)  # removed df_U
    frame = Frame(gb_dir / csv_name, df)
    return frame


def create_csv_with_observed(gp_path: Union[str, Path], gb_destination: str, predicted_outputs: Tuple[NP.Matrix, NP.Matrix, NP.Tensor3],
                             stand_inputs: NP.Matrix, rotated_inputs: NP.Matrix, stand_observed_outputs: NP.Vector, k: int = 0,
                             csv_name: str = "__predictions__.csv"):
    """ Saves the inputs, rotated inputs, predictions and observed output into a CSV file.

    Args:
        gp_path: The path that takes you into the trained Gaussian process' Split folder where the Fold can be found.
        gb_destination: The name of the new folder where the GB will be copied and predictions made e.g. "ROM_Predictions".
        predicted_outputs: A pair (predictive_mean, predictive_std) of (N,1) numpy arrays.
        stand_inputs: The standardized input values - an (N,M) numpy array, consisting of N test inputs, each of dimension M.
        rotated_inputs: The rotated inputs that are predicted on.
        stand_observed_outputs: A NP.Vector of standardized observed outputs from the inputs.
        k: The fold that contains the data that has been used to train the GP's.
        csv_name: The name of the csv file that has been created. Default is "__predictions__".
    Returns: A frame containing the inputs, the rotated inputs, the predictions and the observed outputs.

    """
    gp_path = Path(gp_path)
    split_store = Store(gp_path)
    fold = Fold(split_store, k)
    gb_dir = fold.dir / gb_destination
    mean_prediction_T = predicted_outputs[0]
    std_prediction_T = np.sqrt(predicted_outputs[1])
    df_X = pd.DataFrame(stand_inputs)
    inputs_label = ['x' + str(i - 1) for i in np.arange(1, len(df_X.columns) + 1)]
    inputs_dict = dict(zip(df_X.columns, inputs_label))
    df_X = df_X.rename(columns=inputs_dict)
    list_input_header = list(df_X.columns.values)
    for idx, val in enumerate(list_input_header):
        list_input_header[idx] = 'Input'
    df_X.columns = pd.MultiIndex.from_tuples(zip(list_input_header, df_X.columns))
    """df_U = pd.DataFrame(Rotated_Inputs)
    rotated_label = ['u' + str(i - 1) for i in np.arange(1, len(df_U.columns) + 1)]
    rotated_dict = dict(zip(df_U.columns, rotated_label))
    df_U = df_U.rename(columns=rotated_dict)
    list_input_header = list(df_U.columns.values)
    for idx, val in enumerate(list_input_header):
        list_input_header[idx] = 'Input'
    df_U.columns = pd.MultiIndex.from_tuples(zip(list_input_header, df_U.columns))"""
    df_mean = pd.DataFrame(mean_prediction_T)
    mean_label = ['Predicted_Mean']
    mean_dict = dict(zip(df_mean.columns, mean_label))
    df_mean = df_mean.rename(columns=mean_dict)
    df_std = pd.DataFrame(std_prediction_T)
    std_label = ['Predicted_STD']
    std_dict = dict(zip(df_std.columns, std_label))
    df_std = df_std.rename(columns=std_dict)
    df_Y = pd.DataFrame(stand_observed_outputs)
    Y_label = ['Observed_Output']
    Y_dict = dict(zip(df_Y.columns, Y_label))
    df_Y = df_Y.rename(columns=Y_dict)
    df_outputs = pd.concat([df_mean, df_std, df_Y], axis=1)
    list_output_header = list(df_outputs.columns.values)
    for idx, val in enumerate(list_output_header):
        list_output_header[idx] = 'Output'
    df_outputs.columns = pd.MultiIndex.from_tuples(zip(list_output_header, df_outputs.columns))
    df = pd.concat([df_X, df_outputs], axis=1)  # removed df_U
    frame = Frame(gb_dir / csv_name, df)
    return frame


def collect_predictions(store: Store, folder_name: str, csv_name: str, is_split: bool = True) -> Union[List[Tuple[int, Path]],
                                                                                                       List[Tuple[None, Path]]]:
    """Service routine to instantiate the collection of prediction results.

        Args:
            csv_name:
            store: The Store containing the global dataset to be analyzed.
            folder_name: The name of the folder where the results are being collected from.
            is_split: True or False, whether splits have been used in the model.
        Returns: The split directories collected.
    """
    final_frame = frame = None
    if is_split:
        final_destination = store.dir / folder_name
        final_destination.mkdir(mode=0o777, parents=True, exist_ok=True)
        splits = store.splits
    else:
        final_destination = None
        splits = [(None, store.dir)]
    for split in splits:
        split_store = Store(split[-1])
        K = split_store.meta['K']
        destination = split_store.dir / folder_name
        destination.mkdir(mode=0o777, parents=True, exist_ok=True)
        for k in range(K):
            fold = Fold(split_store, k)
            source = (fold.dir / folder_name) / csv_name
            # PARAMETERS = {'sep': ',',
            #              'header': [0],
            #              'index_col': 0, }
            result = Frame(source).df
            result.insert(0, "Fold", np.full(result.shape[0], k), True)
            # out = result.iloc[:, -1]
            # std = result.iloc[:, -2]
            # mean = result.iloc[:, -3]
            # result.iloc[:, -1] = (out - mean) / std
            if k == 0:
                frame = Frame(destination / csv_name, result.copy(deep=True))
            else:
                frame.df = pd.concat([frame.df, result.copy(deep=True)], axis=0, ignore_index=False)
        frame.write()
        if is_split:
            result = frame.df
            # rep = dict([(result['Predicted_Mean'].columns[0], "Observed_Output")])
            # result.rename(columns=rep, level=1, inplace=True)
            result.insert(0, "Split", np.full(result.shape[0], split[0]), True)
            # result = result.reset_index()
            if split[0] == 0:
                final_frame = Frame(final_destination / csv_name, result.copy(deep=True))
            else:
                final_frame.df = pd.concat([final_frame.df, result.copy(deep=True)], axis=0, ignore_index=True)
    if is_split:
        final_frame.write()
    return splits


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
    df2.insert(loc, "X", membership)
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
    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\1-Jan")
    # BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
    Source = 'Test_Fold_Unclustered.csv'
    Stand_Source = "Test_Fold_Unclustered.csv"
    Clustered_Source = 'K_Clustered_Test_Fold.csv'
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    STORE_NAME = "GP-K_Clustering-5folds"
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("1 -- Predictions finsihed")

    """BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\2-Feb")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("2 -- Predictions finished")

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\3-Mar")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("3 -- Predictions finished")

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\4-Apr")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("4 -- Predictions finished")

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\5-May")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("5 -- Predictions finished")

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\6-Jun")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("6 -- Predictions finished")

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\7-Jul")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("7 -- Predictions finished")

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\8-Aug")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("8 -- Predictions finished")

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\9-Sep")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("9 -- Predictions finished")

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\10-Oct")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("10 -- Predictions finished")

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\11-Nov")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("11 -- Predictions finished")

    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing3\\12-Dec")
    k_cluster_step(BASE_PATH, Source, Stand_Source, Clustered_Source)
    # STORE = store_and_fold("x1_z1_with_dummy.csv", STORE_NAME, 5, is_split=False)
    STORE = Store(BASE_PATH / STORE_NAME)
    x, y = test_data(gp_path=Path(BASE_PATH / STORE_NAME), test_data_file=Clustered_Source, standardized=False, k=0,
                     standard_data_file="__standard__.csv")
    predicted_outputs = predict(gp_path=Path(BASE_PATH / STORE_NAME), gb_source="ExponentialQuadratic-ARD",
                                gb_destination="Forecasts", rotated_inputs=x, Us_taken=-1, k=0)
    create_csv_with_observed(gp_path=Path(BASE_PATH / STORE_NAME), gb_destination="Forecasts", predicted_outputs=predicted_outputs,
                             stand_inputs=x, stand_observed_outputs=y, csv_name="__forecasts__.csv", rotated_inputs=x)
    collect_predictions(store=STORE, folder_name="Forecasts", csv_name="__forecasts__.csv", is_split=False)
    print("12 -- Predictions finished")"""

