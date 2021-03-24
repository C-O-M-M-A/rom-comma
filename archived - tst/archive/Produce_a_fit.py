""" Contains the functions to be used to produce a fit from a taught Gaussian process or ROM.

**Contents**:
    **Input_Grid**: A function that returns a grid of known inputs -- Can be computationally expensive for high dimensions.

    **Generate_Inputs**: A function used to generate many inputs.

    **Test_Data**: Loads test_data that can be used as inputs for predictions and to test predictions.

    **Rotate_Inputs**: Rotates the standardized inputs by Theta ready to produce a fit.

    **Predict**: Prediction using a GaussianBundle.

    **Create_CSV**: Saves the inputs, rotated inputs and predictions into a CSV file.

    **Create_CSV_with_Observed**: Saves the inputs, rotated inputs, predictions and observed output into a CSV file.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from ROMCOMMA.tst.romcomma.data import Store, Fold, Frame
import ROMCOMMA.tst.romcomma.model as model
from ROMCOMMA.tst.romcomma.typing_ import NP, Union, Tuple
from shutil import rmtree, copytree
from pyDOE import lhs
from scipy.stats.distributions import norm
from sklearn.metrics import r2_score
import time


def Input_Grid(bound: float = 1.0, n_samples: int = 3, dim: int = 2) -> NP.Matrix:
    """ A function to generate a mesh grid of inputs that can be used to produce a fit.

    Args:
        bound: The upper and lower bound in which the inputs can be.
        n_samples: The number of values between the upper and lower bound.
        dim: The dimensions of an input grid.
    Returns: A NP.matrix of size (dim x n_samples**dim)
    """
    space = np.linspace(-bound, bound, n_samples)
    a_tuple = (space,)*dim
    mesh = np.ravel(np.meshgrid(*a_tuple))
    mesh.shape = (dim, n_samples**dim)
    Stand_Inputs = np.transpose(mesh)
    return Stand_Inputs


def Generate_Inputs(M: int, N: int, mean: float = 0, std: float = 0.5) -> NP.Matrix:
    """ Generates a large number of inputs.

    Args:
        M: The dimensions of the inputs.
        N: The number of samples / amount of inputs generated.
        mean: The samples will be standardized around this mean. The default is 0.
        std: The samples will be standardized by this standard deviation. 95% of the samples will have be within 2 stdv's from the mean.
    Returns: A NP.matrix of size (M x N)
    """
    Stand_Inputs = norm(loc=mean, scale=std).ppf(lhs(M, samples=N, criterion='cm'))
    return Stand_Inputs


def Test_Data(gp_path: Union[str, Path], test_data_file: str, standard_data_file: str) -> Tuple[NP.Matrix, NP.Vector]:
    """ Loads and saves the test data as standardized inputs (X) and standardized observed outputs (Y).

    Args:
        gp_path: Path to the folder (inside a split) where the test data and standard data files are stored.
        test_data_file: The name of the test data file, e.g. 'test_data.csv'.
        standard_data_file: The name of the standard data file which contains the mean and standard deviation used to standardize the test data.
    Returns: A tuple containing the NP.Matrix of inputs (X) and the NP.Vector of outputs (Y).
    """
    GP_PATH = Path(gp_path)
    test_data = pd.read_csv(GP_PATH / test_data_file, index_col=0)
    mean_std = pd.read_csv(GP_PATH / standard_data_file, index_col=0)
    Dataset = test_data.values
    AVG_STD = mean_std.values
    Stand_Dataset = (Dataset - AVG_STD[0]) / AVG_STD[1]
    Stand_Inputs = Stand_Dataset[:, :-1]
    Stand_Observed_Outputs = Stand_Dataset[:, -1]
    return Stand_Inputs, Stand_Observed_Outputs


def Rotate_Inputs(gp_path: Union[str, Path], Stand_Inputs: NP.Matrix, Us_taken: int, k: int = 0) -> NP.Matrix:
    """ Rotates the standardized inputs by theta to produce the rotated inputs that can be used when predicting with a ROM.

    Args:
        gp_path: Path to a model.GaussianBundle. The extension of this filename is the number of input dimensions M.
            An extension of 0 or a missing extension means full order, taking M from the training data.
        Stand_Inputs: The standardized input values - an (N,M) numpy array, consisting of N test inputs, each of dimension M.
        Us_taken: The amount of Us taken from the ROM to be used in predicting.
        k: The fold that contains the data that has been used to train the GP's.
    Returns: The rotated inputs, U, that can be used for predicting using a ROM - a numpy array of dimensions (N x Xs_taken).
    """
    GP_PATH = Path(gp_path)
    Mu = Us_taken
    Split_Store = Store(GP_PATH)
    fold = Fold(Split_Store, k)
    fold_dir = fold.dir
    sobol = fold_dir / "ROM.optimized" / "sobol"
    theta_T = np.transpose(Frame(sobol / "Theta.csv", csv_parameters={'header': [0]}).df.values)
    k = int(fold_dir.suffix[1:])
    M = Fold(fold_dir.parent, k, Mu).M +1
    if 0 < Mu < M:
        Rotated_Inputs = Stand_Inputs @ theta_T[:, 0:Mu]
    else:
        Rotated_Inputs = Stand_Inputs @ theta_T
    return Rotated_Inputs


def Predict(gp_path: Union[str, Path], gb_source: str, gb_destination: str, Rotated_Inputs: NP.Matrix, Us_taken: int ,k: int = 0) -> Tuple[NP.Vector, NP.Vector]:
    """ Prediction using a GaussianBundle.

    Args:
        gp_path: The path that takes you into the trained Gaussian process' Split folder where the Fold can be found.
        gb_source: The name of the trained GB e.g. "ROM.optimized" or "ARD".
        gb_destination: The name of the new folder where the GB will be copied and predictions made e.g. "ROM_Predictions".
        Rotated_Inputs: The rotated inputs that are predicted on.
        Us_taken: The amount of Us taken from the ROM to be used in predicting.
        k: The fold that contains the data that has been used to train the GP's.
    Returns: A pair (predictive_mean, predictive_std) of (N,1) numpy arrays.

    """
    GP_PATH = Path(gp_path)
    Split_Store = Store(GP_PATH)
    fold = Fold(Split_Store, k, Us_taken)
    gb_dir = fold.dir / gb_destination
    rmtree(gb_dir, ignore_errors=True)
    copytree(src=fold.dir / gb_source, dst=gb_dir)
    gb = model.mygpy.GaussianBundle(fold=fold, name=gb_destination, parameters=None, overwrite=True, reset_log_likelihood=False)
    Predicted_Output = gb.predict(Rotated_Inputs)
    return Predicted_Output


def Create_CSVs(gp_path: Union[str, Path], gb_destination: str, Predicted_Outputs: Tuple[NP.Vector, NP.Vector], Stand_Inputs: NP.Matrix,
                Rotated_Inputs: NP.Matrix, k: int = 0, csv_name: str = "__predictions__"):
    """ Saves the inputs, rotated inputs and predictions into a CSV file.

    Args:
        gp_path: The path that takes you into the trained Gaussian process' Split folder where the Fold can be found.
        gb_destination: The name of the new folder where the GB will be copied and predictions made e.g. "ROM_Predictions".
        Predicted_Outputs: A pair (predictive_mean, predictive_std) of (N,1) numpy arrays.
        Stand_Inputs: The standardized input values - an (N,M) numpy array, consisting of N test inputs, each of dimension M.
        Rotated_Inputs: The rotated inputs that are predicted on.
        k: The fold that contains the data that has been used to train the GP's.
        csv_name: The name of the csv file that has been created. Default is "__predictions__".
    Returns: A frame containing the inputs, the rotated inputs and predictions for each split directory.

    """
    GP_PATH = Path(gp_path)
    Split_Store = Store(GP_PATH)
    fold = Fold(Split_Store, k)
    gb_dir = fold.dir / gb_destination
    mean_prediction_T = np.transpose(Predicted_Outputs[0, :, [0]])
    std_prediction_T = np.sqrt(np.transpose(Predicted_Outputs[1, :, [0]]))
    df_X = pd.DataFrame(Stand_Inputs)
    inputs_label = ['x' + str(i - 1) for i in np.arange(1, len(df_X.columns) + 1)]
    inputs_dict = dict(zip(df_X.columns, inputs_label))
    df_X = df_X.rename(columns=inputs_dict)
    DF_U = pd.DataFrame(Rotated_Inputs)
    rotated_label = ['u' + str(i - 1) for i in np.arange(1, len(DF_U.columns) + 1)]
    rotated_dict = dict(zip(DF_U.columns, rotated_label))
    DF_U = DF_U.rename(columns=rotated_dict)
    df_mean = pd.DataFrame(mean_prediction_T)
    mean_label = ['Predicted_Mean']
    mean_dict = dict(zip(df_mean.columns, mean_label))
    df_mean = df_mean.rename(columns=mean_dict)
    df_std = pd.DataFrame(std_prediction_T)
    std_label = ['Predicted_STD']
    std_dict = dict(zip(df_std.columns, std_label))
    df_std = df_std.rename(columns=std_dict)
    df = pd.concat([df_X, DF_U, df_mean, df_std], axis=1)
    frame = Frame(gb_dir / csv_name, df)
    return frame


def Create_CSV_with_Observed(gp_path: Union[str, Path], gb_destination: str, Predicted_Outputs: Tuple[NP.Vector, NP.Vector], Stand_Inputs: NP.Matrix,
                             Rotated_Inputs: NP.Matrix, Stand_Observed_Outputs: NP.Vector, k: int = 0, csv_name: str = "__predictions__"):
    """ Saves the inputs, rotated inputs, predictions and observed output into a CSV file.

    Args:
        gp_path: The path that takes you into the trained Gaussian process' Split folder where the Fold can be found.
        gb_destination: The name of the new folder where the GB will be copied and predictions made e.g. "ROM_Predictions".
        Predicted_Outputs: A pair (predictive_mean, predictive_std) of (N,1) numpy arrays.
        Stand_Inputs: The standardized input values - an (N,M) numpy array, consisting of N test inputs, each of dimension M.
        Rotated_Inputs: The rotated inputs that are predicted on.
        Stand_Observed_Outputs: A NP.Vector of standardized observed outputs from the inputs.
        k: The fold that contains the data that has been used to train the GP's.
        csv_name: The name of the csv file that has been created. Default is "__predictions__".
    Returns: A frame containing the inputs, the rotated inputs, the predictions and the observed outputs.

    """
    GP_PATH = Path(gp_path)
    Split_Store = Store(GP_PATH)
    fold = Fold(Split_Store, k)
    gb_dir = fold.dir / gb_destination
    mean_prediction_T = np.transpose(Predicted_Outputs[0, :, [0]])
    std_prediction_T = np.sqrt(np.transpose(Predicted_Outputs[1, :, [0]]))
    df_X = pd.DataFrame(Stand_Inputs)
    inputs_label = ['x' + str(i - 1) for i in np.arange(1, len(df_X.columns) + 1)]
    inputs_dict = dict(zip(df_X.columns, inputs_label))
    df_X = df_X.rename(columns=inputs_dict)
    DF_U = pd.DataFrame(Rotated_Inputs)
    rotated_label = ['u' + str(i - 1) for i in np.arange(1, len(DF_U.columns) + 1)]
    rotated_dict = dict(zip(DF_U.columns, rotated_label))
    DF_U = DF_U.rename(columns=rotated_dict)
    df_mean = pd.DataFrame(mean_prediction_T)
    mean_label = ['Predicted_Mean']
    mean_dict = dict(zip(df_mean.columns, mean_label))
    df_mean = df_mean.rename(columns=mean_dict)
    df_std = pd.DataFrame(std_prediction_T)
    std_label = ['Predicted_STD']
    std_dict = dict(zip(df_std.columns, std_label))
    df_std = df_std.rename(columns=std_dict)
    df_Y = pd.DataFrame(Stand_Observed_Outputs)
    Y_label = ['Observed_Output']
    Y_dict = dict(zip(df_Y.columns, Y_label))
    df_Y = df_Y.rename(columns=Y_dict)
    df = pd.concat([df_X, DF_U, df_mean, df_std, df_Y], axis=1)
    frame = Frame(gb_dir / csv_name, df)
    return frame


if __name__ == '__main__':
    start = time.time()
    Base_Path = Path('X:\\comma_group1\\Rom\\dat\\AaronsTraining\\MOF\\MOF_Data_Reduced_6D\\MOF_Rotated_1.1')
    store = Store(Base_Path)
    splits = store.splits
    for split_dir in splits:
        split_start = time.time()
        split = Store(split_dir)
        GP_Path = Path(Base_Path / split_dir)
        Mu = 6
        X, Y = Test_Data(gp_path=GP_Path, test_data_file="__test_data__.csv", standard_data_file="__test_standard__.csv")
        U = Rotate_Inputs(gp_path=GP_Path, Stand_Inputs=X, Us_taken=Mu)
        Prediction = Predict(gp_path=GP_Path, gb_source="ROM.optimized", gb_destination="ROM.Predictions", Rotated_Inputs=U, Us_taken=Mu)
        predict_frame = Create_CSV_with_Observed(gp_path=GP_Path, gb_destination="ROM.Predictions", Predicted_Outputs=Prediction,
                                                 Stand_Inputs=X, Rotated_Inputs=U, Stand_Observed_Outputs=Y, csv_name="__predictions.Mu.6__")
        print('Goodness of fit = ', r2_score(Y, np.transpose(Prediction[0, :, [0]])))
        split_time_mins = (time.time() - split_start) / 60
        print(split_dir.name, "has finished in {:.2f} minutes.".format(split_time_mins))
    print()
    time_mins = (time.time() - start) / 60
    print("It took {:.2f} minutes to execute this code.".format(time_mins))
