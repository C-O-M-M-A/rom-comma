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

""" Encapsulates data storage structures.

Contents:
    **Frame**: The class that encapsulates a dataframe which is backed by a CSV file.
    **Store**: The class that specifies the directory containing the global dataset to be analyzed.
    **Fold(Store)**: class. The class that specifies the directory containing the datasets for modelling and for testing.
"""
from romcomma.typing_ import Cls, Callable, PathLike, ZeroOrMoreInts, List, Tuple, Union
from copy import deepcopy
from itertools import chain
from random import shuffle
from pathlib import Path
from numpy import NaN, append
from pandas import DataFrame, Series, read_csv, concat
from enum import IntEnum, auto
import json


class Frame:
    """ Encapsulates a DataFrame (df) backed by a csv file."""

    CSV_PARAMETERS = {'sep': ',',
                      'header': [0, 1],
                      'index_col': 0, }

    @classmethod
    def empty_fr(cls):
        """ The empty frame."""
        return cls(Path(), DataFrame())

    @property
    def csv(self) -> Path:
        """ The csv csv."""
        return self._csv

    @property
    def empty(self) -> bool:
        """ The csv csv."""
        return 0 == len(self._csv.parts)

    def write(self):
        """ Write to csv, according to Frame.ORIGIN_CSV_PARAMETERS."""
        if self.empty:
            raise FileNotFoundError("Cannot write an empty frame.")
        self.df.to_csv(self._csv, sep=Frame.CSV_PARAMETERS['sep'], index=True)

    # noinspection PyDefaultArgument
    def __init__(self, csv: PathLike = Path(), df: DataFrame = DataFrame(),
                 csv_parameters: dict = CSV_PARAMETERS):
        """ Initialize data.Frame.

        Args:
            csv: The csv file. Required.
            df: The initial data. If this is empty, it is read from ``csv``, otherwise it overwrites
                (or creates) ``csv``.
            csv_parameters: The parameters used for reading, as detailed in
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html.
                *This is not relevant to writing, which is standardized using Frame.ORIGIN_CSV_PARAMETERS*.
        """
        self._csv = Path(csv)
        if self.empty:
            self.df = df
            if not self.df.empty:
                raise FileNotFoundError("csv is an empty path, but df is not an empty DataFrame.")
        elif df.columns.empty:
            _csv_parameters = {**Frame.CSV_PARAMETERS, **csv_parameters}
            self.df = read_csv(self._csv, **_csv_parameters)
        else:
            self.df = df
            self.write()


# noinspection PyDefaultArgument
class Store:
    """ A Store is defined as a dir containing a ``__data__.csv`` file and a ``__meta__.json`` file.
    These files specify the global dataset to be analyzed.
    """

    class Standard:
        """ Encapsulates specifications for standardizing data as ``classmethods``."""

        Specification = Callable[[DataFrame], DataFrame]

        @staticmethod
        def _mean(df: DataFrame) -> Series:
            mean = df.mean()
            mean.name = 'mean'
            return mean

        @staticmethod
        def _stack_as_rows(top: Series, bottom: Series):
            return concat([top, bottom], axis=1).T

        # noinspection PyUnusedLocal
        @classmethod
        def none(cls, df: DataFrame) -> DataFrame:
            """ A function which returns a dataframe from nothing."""
            return DataFrame()

        @classmethod
        def mean_and_range(cls, df: DataFrame) -> DataFrame:
            """ Returns the mean and the range of data.
                Args:
                    **df**: The initial data
                """
            scale = df.max() - df.min()
            scale.name = 'range'
            result = cls._stack_as_rows(cls._mean(df), scale)
            return result

        @classmethod
        def mean_and_std(cls, df: DataFrame) -> DataFrame:
            """ Returns the mean and the standard deviation of data.
                Args:
                    **df**: The initial data
                """
            scale = df.std()
            scale.name = 'std'
            return cls._stack_as_rows(cls._mean(df), scale)

    META = {'csv_parameters': Frame.CSV_PARAMETERS, 'standard': Standard.none.__name__, 'data': {},
            'K': 0, 'shuffled before folding': False}

    @property
    def dir(self) -> Path:
        """ The function that defines the directory resulting in a path."""
        return self._dir

    @property
    def data_csv(self) -> Path:
        """ The function that defines the directory to the data file."""
        return self._dir / "__data__.csv"

    @property
    def data(self) -> Frame:
        """ The function that defines the frame consisting of the data file."""
        self._data = Frame(self.data_csv) if self._data is None else self._data
        return self._data

    @property
    def meta_json(self) -> Path:
        """ The function that defines the __meta__.json file."""
        return self._dir / "__meta__.json"

    @property
    def meta(self) -> dict:
        """ The function that defines the dict for meta."""
        return self._meta

    @property
    def N(self) -> int:
        """ The function that returns the number of datapoints in the data file."""
        return self._meta['data']['N']

    @property
    def M(self) -> int:
        """ The function that returns the number of inputs in the data file."""
        return self._meta['data']['M']

    @property
    def L(self) -> int:
        """ The function that returns the number of outputs in the data file"""
        return self._meta['data']['L']

    @property
    def K(self) -> int:
        """ The function that returns the number of folds."""
        return self._meta['K']

    def _read_meta_json(self) -> dict:
        """ The function that opens and reads the meta_json object."""
        # noinspection PyTypeChecker
        with open(self.meta_json, mode='r') as file:
            return json.load(file)

    def _write_meta_json(self):
        """ The function that writes the meta object as a json file."""
        # noinspection PyTypeChecker
        with open(self.meta_json, mode='w') as file:
            json.dump(self._meta, file, indent=8)

    @property
    def standard_csv(self) -> Path:
        """ The function that returns the paths for the standardised CSV data file."""
        return self._dir / "__standard__.csv" if self.standardized else Path()

    @property
    def pre_standard_data_csv(self) -> Path:
        """ The function that returns the paths for the pre-standardised CSV data file."""
        return self._dir / "__pre-standard_data__.csv"

    @property
    def standard(self) -> Frame:
        """ The function that encapsulates the Frame for the standard parameters."""
        self._standard = Frame(self.standard_csv) if self._standard is None else self._standard
        return self._standard

    @property
    def standardized(self) -> bool:
        """ The function that returns True or False whether the data is standardized."""
        return self._meta['standard'] != Store.Standard.none.__name__

    def create_standardized_frame(self, csv: PathLike, df: DataFrame) -> Frame:
        """ Overwrite ``df`` with its standardized version, saving to csv.

        Args:
            df: Data to be standardized.
            csv: Locates the return Frame.
        Returns: A Frame written to ``csv`` containing ``df``, standardized.
        """
        if self.standardized:
            df = (df - self.standard.df.iloc[0]) / self.standard.df.iloc[1]
        return Frame(csv, df)

    def standardize(self, standard: Standard.Specification, save_non_standard: bool = True) -> Frame:
        """ Standardize this data.Store, and update ``self.meta``.
        If standard==``Standard.none``, nothing else is done.
        Otherwise, standard (a function member of Standard.Specification) is applied to
        ``self.data.df``, standardizing it, and writing the files ``[self.standard_csv]``,
        ``[self.data_csv]`` and, optionally, ``[self.pre_standard_data_csv]``.

        Args:
            standard: Specification of Standard: ``Standard.none``, ``Standard.mean_and_range``
                or ``Standard.mean_and_std``.
            save_non_standard: Whether to save ``[self.pre_standard_data_csv]``.
        Returns: ``self.standard``.
        """
        self._meta['standard'] = standard.__name__
        self._write_meta_json()
        self._standard = Frame(self.standard_csv, standard(self.data.df))
        if self.standardized:
            if save_non_standard:
                self.data_csv.replace(self.pre_standard_data_csv)
            self._data = self.create_standardized_frame(self.data_csv, self._data.df)
        return self._standard

    def fold_dir(self, k: int) -> Path:
        """ Returns the path containing each fold between 0 and K.

        Args:
            k: The fold which the function is creating the path for.
        Raises:
            IndexError: if k is not between or equal to 0 and K.
        """
        if 0 <= k < self.K:
            return self.dir / "fold.{k:d}".format(k=k)
        else:
            raise IndexError("Requested fold {k:d} of a {K:d}-fold data.Store".format(k=k, K=self.K))

    def _K_folds_update(self, K: int, shuffled_before_folding: bool):
        """ The function that updates and writes the meta fold file.

        Args:
            K: The total number of folds.
            shuffled_before_folding: True or False, whether the data should be shuffled before folding.
        Returns:
            A meta json file containing the parameters for folding the data file.
        """
        self._meta.update({'K': K,
                           'shuffled before folding': shuffled_before_folding})
        self._write_meta_json()

    def _data_update(self):
        """ Takes the different headings in the dataframe to update the data with the number of data points N, inputs M, and outputs L"""
        self._meta.update({'data': {'X_heading': self._data.df.columns.values[0][0],
                                    'Y_heading': self._data.df.columns.values[-1][0]}})
        self._meta['data'].update({'N': self.data.df.shape[0], 'M': self.X.shape[1],
                                   'L': self.Y.shape[1]})

    @property
    def X(self) -> DataFrame:
        """ The function defines the 'X_heading' in the dataframe as X."""
        return self.data.df[self._meta['data']['X_heading']]

    @property
    def Y(self) -> DataFrame:
        """ The function defines the 'Y_heading' in the dataframe as Y."""
        return self.data.df[self._meta['data']['Y_heading']]

    def drop(self, columns: List[Tuple[str, str]]):
        """ Drop columns from ``self.data``.
        Args:
            columns: A list of pairs (tuples) of strings heading a column - e.g. ('Input','X')
        """
        self.data.df.drop(columns, axis=1, inplace=True)
        self._data_update()
        self._write_meta_json()
        self._data.write()

    # noinspection PyUnresolvedReferences,PyUnresolvedReferences
    def split(self):
        """ A function that instantiates the splitting of a dataframe to L splits, returning L amounts of new Frame's."""
        for l in range(self.L):
            destination = ((self.dir.parent / "split.{0:d}".format(l)) / self.dir.name if self.__class__ == Fold
                           else self.dir / "split.{0:d}".format(l))
            if not destination.exists():
                destination.mkdir(mode=0o777, parents=True, exist_ok=False)
            indices = append(range(self.M), self.M + l)
            data = self.data.df.take(indices, axis=1, is_copy=True)
            Frame(destination / self.data_csv.name, data)
            meta = deepcopy(self._meta)
            meta['data']['L'] = 1
            with open(destination / self.meta_json.name, mode='w') as file:
                json.dump(meta, file, indent=8)
            if self.standardized:
                standard = self.standard.df.take(indices, axis=1, is_copy=True)
                Frame(destination / self.standard_csv.name, standard)
            if self.__class__ == Fold:
                test = self._test.df.take(indices, axis=1, is_copy=True)
                Frame(destination / self.test_csv.name, test)
            else:
                for k in range(self.K):
                    fold = Fold(self, k)
                    fold.split()

    @property
    def splits(self) -> List[Path]:
        """ A function to define a list of paths for each split."""
        return list(self.dir.glob("split.[0-9]*"))

    class InitMode(IntEnum):
        """ A function to create enumerated constants that are also subclasses of int. It replaces the instances with an appropriate value"""
        READ_META_ONLY = auto()
        READ = auto()
        CREATE_AND_OVERWRITE = auto()
        CREATE_NOT_OVERWRITE = auto()

    def __init__(self, dir_: PathLike, init_mode: InitMode = InitMode.READ):
        """ Initialize data.Store.

        Args:
            dir_: The location (directory) of the data.Store.
            init_mode: The mode to initialize with, variations on READ (an existing Store) and CREATE (a new one).
        Raises:
            FileNotFoundError: Missing ``self.meta_json``.
            FileNotFoundError: Missing ``self.data_csv``.
            FileExistsError: Attempt to overwrite existing ``dir_`` when ``_overwrite`` was Falsified
                by  ``Store.from_df(dir_, ..., overwrite=False)``.
        """
        self._dir = Path(dir_)
        self._data = None
        self._standard = None
        if init_mode <= Store.InitMode.READ:
            self._meta = self._read_meta_json()
            if init_mode is Store.InitMode.READ:
                self._data = Frame(self.data_csv)
        else:
            if self._dir.exists() and init_mode is Store.InitMode.CREATE_NOT_OVERWRITE:
                raise FileExistsError(("I will not overwrite {0} unless you call " +
                                       "Store.from_df({0}, ..., overwrite=True)").format(self._dir))
            self._dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    @classmethod
    def from_df(cls, dir_: PathLike, df: DataFrame, meta: dict = META, overwrite: bool = False) -> Cls:
        """ Create a data.Store.

        Args:
            dir_: The location (directory) of the data.store
            df: The data to store in ``[Return].data_csv``.
            meta: The metadata to store in ``[Return].meta_json``.
            overwrite: Whether to overwrite an existing data.Store.
        Returns: A new data.Store.
        """
        _init_mode = Store.InitMode.CREATE_AND_OVERWRITE if overwrite else Store.InitMode.CREATE_NOT_OVERWRITE
        store = Store(dir_, _init_mode)
        store._meta = {**Store.META, **meta}
        store._data = Frame(store.data_csv, df)
        store._data_update()
        store.standardize(Store.Standard.none)
        return store

    ORIGIN_CSV_PARAMETERS = {'skiprows': None, 'index_col': None}

    @classmethod
    def from_csv(cls, dir_: PathLike, csv: PathLike, csv_parameters: dict = ORIGIN_CSV_PARAMETERS,
                 skiprows: ZeroOrMoreInts = None, meta: dict = META, overwrite: bool = False) -> Cls:
        """ Create a data.Store.

        Args:
            dir_: The location (directory) of the data.store
            csv: File containing the data to store in ``[Return].data_csv``.
            csv_parameters: A ``kwargs``-like ``dict`` of parameters passed to the csv reader, as detailed in
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html.
            skiprows: The rows of csv to skip while reading, a convenience update to ``csv_parameters``.
            meta: The metadata to store in ``[Return].meta_json``.
            overwrite: Whether to overwrite any existing dir at ``dir``.
        Returns: A new data.Store.
        """
        csv = Path(csv)
        _meta = {**Store.META, **meta}
        if skiprows is not None:
            csv_parameters.update({'skiprows': skiprows})
        data = Frame(csv, csv_parameters=csv_parameters)
        _meta['origin'] = {'csv': str(csv.absolute()),
                           'csv_parameters': csv_parameters}
        return cls.from_df(dir_, data.df, _meta, overwrite)


class Fold(Store):
    """ A data.Fold is defined as a dir containing a ``__data__.csv``, a ``__test__.csv`` file
    and a ``__meta__.json`` file. These files specify the datasets ``__data__.csv`` for modelling and ``__test__.csv``
    for testing. In other words, a data.Fold is a data.Store equipped with a test frame backed by ``__test__.csv``.
    """

    @property
    def test_csv(self) -> Path:
        """ The path directory for the test csv."""
        return self._dir / "__test__.csv"

    @property
    def M(self) -> int:
        """ The amount of inputs in the file."""
        return self._Xs_taken if self._Xs_taken else super().M

    @property
    def test(self) -> Frame:
        """ The Function defining the Frame containing the test data"""
        return self._test

    @property
    def test_X(self) -> DataFrame:
        """ Returns the test frame with labelled axis for the inputs."""
        return self._test.df[self._meta['data']['X_heading']] if self._Xs_taken else self._test.df[self._meta['data']['X_heading']]

    @property
    def X(self) -> DataFrame:
        """ Returns the label for the X-axis."""
        return super().X.iloc[:, 0:self._Xs_taken] if self._Xs_taken else super().X

    @property
    def test_Y(self) -> DataFrame:
        """ Returns the test frame with labelled axis for the outputs. """
        return self._test.df[self._meta['data']['Y_heading']]

    def set_test_data(self, df: DataFrame):
        """ Instantiates the creation of the test dataframe.

        args:
            **df**: The initial data.
        Returns:
             The standardised dataframe used for testing.
        """
        self._test = self.create_standardized_frame(self.test_csv, df)

    def __init__(self, parent: Union[Store, PathLike], k: int, Xs_taken: int = -1):
        """ Initialize data.Fold by reading existing files. Creation is handled by classmethods named
            ``data.Fold.from_[???]``.

        Args:
            parent: The parent data.Store, or its (dir) csv.
            k: The index of the fold within ``parent``.
            Xs_taken: The number of X columns used. If not 0 < _Xs_taken< self.M, all columns are used.
        Raises:
            IndexError: Unless ``0 <= k < parent.K``
            FileNotFoundError: Missing ``self.test_csv``.
        """
        if not isinstance(parent, Store):
            parent = Store(parent, Store.InitMode.READ_META_ONLY)
        if not (0 <= k < parent.K):
            raise IndexError("Fold k={k:d} is out of bounds 0 <= k < K = {K:d} in data.Store({parent_dir:s}"
                             .format(k=k, K=parent.K, parent_dir=str(parent.dir)))
        super().__init__(parent.fold_dir(k))
        self._Xs_taken = Xs_taken if 0 < Xs_taken < super().M else 0
        if not self.test_csv.is_file():
            raise FileNotFoundError("This Fold is missing {test_filename:s}.".format(test_filename=self.test_csv))
        self._test = Frame(self.test_csv)

    META = {'parent_dir': "", 'k_fold': -1, 'K': -1}

    @classmethod
    def into_K_folds(cls, parent: Store, K: int, shuffled_before_folding: bool = True,
                     standard: Store.Standard.Specification = Store.Standard.mean_and_std, overwrite: bool = False,
                     replace_empty_test_with_data_: bool = True):
        """ Fold parent into K Folds for testing.

        Args:
            parent: The data.Store to fold into K.
            K: The number of Folds, must be between 1 and ``N`` inclusive.
            shuffled_before_folding: Whether to shuffle the samples before sampling. If ``False``,
                each Fold.test will contain 1 sample from the first ``K`` samples in ``parent.__data__``,
                1 sample from the second ``K`` samples, and so on.
            standard: Specification of Standard: ``Standard.none``, ``Standard.mean_and_range``
                or ``Standard.mean_and_std``.
            overwrite: Whether to overwrite any existing Folds.
            replace_empty_test_with_data_: Whether to replace an empty test file with the training data when K==1.
        Raises:
            ValueError: Unless ``1 <= K <= N``.
        """
        N = len(parent.data.df.index)
        if not (1 <= K <= N):
            raise \
                ValueError("K={K:d} does not lie between 1 and N=len(self.data.df.index)={N:d} inclusive"
                           .format(K=K, N=N))

        for k in range(K, parent.K):
            parent.fold_dir(k).mkdir(mode=0o777, parents=False, exist_ok=overwrite)
            parent.fold_dir(k).rmdir()
        parent._K_folds_update(K, shuffled_before_folding)

        indices = list(range(N))
        if shuffled_before_folding:
            shuffle(indices)

        # noinspection PyUnresolvedReferences,PyUnresolvedReferences,PyUnresolvedReferences
        def _fold_from_indices(_k: int, train: List[int], test: List[int]):
            assert len(train) > 0
            meta = {**Fold.META, **{'parent_dir': str(parent.dir), 'k_fold': _k,
                                    'K': parent.K}}
            fold = Store.from_df(parent.fold_dir(_k), parent.data.df.iloc[train], meta, overwrite=overwrite)
            fold.standardize(standard, False)
            fold.__class__ = cls
            if len(test) < 1:
                if replace_empty_test_with_data_:
                    fold._test = fold.create_standardized_frame(fold.test_csv, parent.data.df.iloc[train])
                else:
                    fold._test = Frame(fold.test_csv,
                                       DataFrame(data=NaN, index=[-1], columns=parent.data.df.columns))
            else:
                fold._test = fold.create_standardized_frame(fold.test_csv, parent.data.df.iloc[test])

        # noinspection PyUnusedLocal
        def _indicators():
            K_blocks = [list(range(K)) for i in range(int(N / K))]
            K_blocks.append(list(range(N % K)))
            for K_range in K_blocks:
                shuffle(K_range)
            return list(chain(*K_blocks))

        if 1 == K:
            _fold_from_indices(_k=0, train=indices, test=[])
        else:
            indicators = _indicators()
            for k in range(K):
                _fold_from_indices(_k=k,
                                   train=[index for index, indicator in zip(indices, indicators) if k != indicator],
                                   test=[index for index, indicator in zip(indices, indicators) if k == indicator])
        return K

    @staticmethod
    def _rename(dir_: Path):
        """ A function that renames any directories called fold_# to fold.#"""
        for p in dir_.iterdir():
            if p.is_dir():
                Fold._rename(p)
                split_name = p.name.split("_")
                if split_name[0] == "fold":
                    p.rename(p.parent / (split_name[0] + "." + split_name[1]))
