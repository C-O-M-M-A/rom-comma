
"""
A function that can be used to make predictions on a test data csv file using either a GP or a ROM or both.

Some terminology on the directories:
Base_Path: the base path is where the initial data is found.
Store: inside the base path there maybe multiple stores depending on how the data has been treated using "store_and_fold".
Split: the next directory will usually be the splits if the data has more than 1 output.
Fold: Each fold will then be found.
Models: The models can be the GP (e.g. "ARD") or a ROM (e.g. "ROM.optimized"). Each rotation of the ROM will be in here too (e.g. "ROM.0").
After that, files are collected together back down these directories.
"""


import time
import numpy as np
import pandas as pd
from shutil import rmtree, copytree
from pathlib import Path
from random import shuffle
from itertools import chain
from .data import Store, Fold, Frame
import .model as model
from .typing_ import NP, Union, Tuple, Sequence, List


def make_predictions(input_source: str, store_name: str, is_split: bool = True, is_standardized: bool = False, shuffle_before_folding: bool = True):
    # Ensure the functions can get to the main directories.
    base_path = Path(BASE_PATH)
    store = Store(BASE_PATH / store_name)
    # Read the input test data file which has two rows of headers and 1 index column.
    # This file should be located in the BASE_PATH with the initial training data.
    input_data = pd.read_csv(base_path / input_source, header=[0, 1], index_col=0)
    # has the data already been standardized?
    if is_standardized is True:
        stand_inputs = input_data.values[:, :]
        stand_inputs_df = pd.DataFrame(stand_inputs)
    else:
        # Read the csv storing the standardised information in STORE
        mean_std = pd.read_csv(store.standard_csv, header=[0, 1], index_col=0)
        # Remove the standard values for the outputs.
        mean_std = mean_std.drop(columns=['Output', 'Output']).values
        # Standardise the input data
        dataset = input_data.values
        mean = mean_std[0].astype(float)
        std = mean_std[1].astype(float)
        stand_inputs= (dataset - mean) / std
        stand_inputs = stand_inputs[:, :]
        stand_inputs_df = pd.DataFrame(stand_inputs)
    K = store.K
    N = len(stand_inputs_df.index)
    # assert 1 <= K <= N, "K={K:d} does not lie between 1 and N=len(stand_inputs_df.index)={N:d} inclusive".format(K=K, N=N)
    indices = list(range(N))
    # looks like I need to follow the inner functions Fold.indicators and Fold.fold_from_indices.
    if shuffle_before_folding:
        shuffle(indices)
    K_blocks = [list(range(K)) for i in range(int(N / K))]
    K_blocks.append(list(range(N % K)))
    for K_range in K_blocks:
        shuffle(K_range)
    indicators = list(chain(*K_blocks))
    for k in range(K):
        train = [index for index, indicator in zip(indices, indicators) if k != indicator]
        test = [index for index, indicator in zip(indices, indicators) if k == indicator]
        assert len(train) > 0
    """
    indicators = _indicators()
    for k in range(K):
        _fold_from_indices(_k=k,
                           train=[index for index, indicator in zip(indices, indicators) if k != indicator],
                           test=[index for index, indicator in zip(indices, indicators) if k == indicator])
        def _fold_from_indices(_k: int, train: List[int], test: List[int]):
            assert len(train) > 0
            meta = {**Fold.DEFAULT_META, **{'parent_dir': str(parent.dir), 'k': _k,
                                    'K': parent.K}}
            fold = Store.from_df(parent.fold_dir(_k), parent.data.df.iloc[train], meta)
            fold.standardize(standard)
            fold.__class__ = cls
            if len(test) < 1:
                if replace_empty_test_with_data_:
                    fold._test = fold.create_standardized_frame(fold.test_csv, parent.data.df.iloc[train])
                else:
                    fold._test = Frame(fold.test_csv,
                                       DataFrame(data=NaN, index=[-1], columns=parent.data.df.columns))
            else:
                fold._test = fold.create_standardized_frame(fold.test_csv, parent.data.df.iloc[test])"""
    return


if __name__ == '__main__':
    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Veysel-Copy")
    STORE_NAME_1 = "10_Folds"
    STORE_NAME_2 = "1_Fold"
    make_predictions(input_source="Input_Data.csv", store_name=STORE_NAME_1, is_split=False)