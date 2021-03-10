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
    **Frame** class encapsulating a pandas DataFrame backed by a CSV file.
    
    **Store** class container for datasets. 
    A Store is defined as a dir containing a ``__data__.csv`` file and a ``__meta__.json`` file.
    
    **Store.Standard** class encapsulating specifications for standardizing data.

    **Fold(Store)** class container for datasets, models and tests.
"""
from romcomma.typing_ import Callable, PathLike, ZeroOrMoreInts, List, Tuple, Union, Dict
from copy import deepcopy
from itertools import chain
from random import shuffle
from pathlib import Path
from numpy import NaN, append
from pandas import DataFrame, Series, read_csv, concat
from enum import IntEnum, auto
import json


class Frame:
    """ Encapsulates a DataFrame (df) backed by a csv file.

    Attributes:
        df: The pandas DataFrame.
    """

    DEFAULT_CSV_KWARGS = {'sep': ',',
                      'header': [0, 1],
                      'index_col': 0, }

    @property
    def csv(self) -> Path:
        """ The csv file."""
        return self._csv

    @property
    def is_empty(self) -> bool:
        """ Defines the empty Frame as that having an empty Path."""
        return 0 == len(self._csv.parts)

    @property
    def shape(self) -> Tuple:
        """ The shape of the DataFrame."""
        return self.df.shape

    def write(self):
        """ Write to csv, according to Frame.DEFAULT_CSV_KWARGS.

        Raises:
            AssertionError: If self.is_empty.
        """
        assert not self.is_empty, "Cannot write when frame.is_empty."
        self.df.to_csv(self._csv, sep=Frame.DEFAULT_CSV_KWARGS['sep'], index=True)

    # noinspection PyDefaultArgument
    def __init__(self, csv: PathLike = Path(), df: DataFrame = DataFrame(), **kwargs):
        """ Initialize Frame.

        Args:
            csv: The csv file. Required.
            df: The initial data. If this is empty, it is read from csv, otherwise it overwrites
                (or creates) csv.
        Keyword Args:
            kwargs: Updates Frame.DEFAULT_CSV_KWARGS for csv reading as detailed in
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html.
                This is not relevant to writing, which just uses Frame.DEFAULT_CSV_KWARGS.

        Raises:
            AssertionError: If csv and df are both empty.
        """
        self._csv = Path(csv)
        if self.is_empty:
            assert df.empty, "csv is an empty path, but df is not an empty DataFrame."
            self.df = df
        elif df.empty:
            self.df = read_csv(self._csv, **{**Frame.DEFAULT_CSV_KWARGS, **kwargs})
        else:
            self.df = df
            self.write()


# noinspection PyDefaultArgument
class Store:
    """ A ``store`` object is defined as a ``store.dir`` containing a ``store.data_csv`` file and a ``store.meta_json`` file.

    A Store may also optionally contain a ``store.standard_csv`` file specifying the standardization which has been applied to the data.

    These files specify the global dataset to be analyzed. This dataset may be further split into Folds contained within the Store.
    """

    class Standard:
        """ Encapsulates Specifications for standardizing data as ``classmethods``.

        A Specification is a function taking an (N,M+L) DataFrame ``df `` (to be standardized) as input and returning a (2,M+L) DataFrame.
        The first row of the return contains (,M+L) ``loc`` values, the second row (,M+L) ``scale`` values.

        Ultimately the Store class standardizes ``df`` to ``(df - loc)/scale``.
        """

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
            """ Standard.Specification for un-standardized data.

            Args:
                df: The data to be standardized.
            Returns: An empty DataFrame.
            """
            return DataFrame()

        @classmethod
        def mean_and_range(cls, df: DataFrame) -> DataFrame:
            """ Standard.Specification.

                Args:
                    df: The data to be standardized.
                Returns: The mean and range of data.
            """
            scale = df.max() - df.min()
            scale.name = 'range'
            result = cls._stack_as_rows(cls._mean(df), scale)
            return result

        @classmethod
        def mean_and_std(cls, df: DataFrame) -> DataFrame:
            """ Standard.Specification.

                Args:
                    df: The data to be standardized.
                Returns: The mean and std (unbiased standard deviation) of data.
            """
            scale = df.std()    # pandas std is unbiased (n-1) denominator.
            scale.name = 'std'
            return cls._stack_as_rows(cls._mean(df), scale)

    class InitMode(IntEnum):
        READ_META_ONLY = auto()
        READ = auto()
        CREATE = auto()

    DEFAULT_META = {'csv_kwargs': Frame.DEFAULT_CSV_KWARGS, 'standard': Standard.none.__name__, 'data': {}, 'K': 0, 'shuffled before folding': False}

    @property
    def dir(self) -> Path:
        """ The Store directory."""
        return self._dir

    @property
    def data_csv(self) -> Path:
        """ The Store data file."""
        return self._dir / "__data__.csv"

    @property
    def meta_json(self) -> Path:
        """ The Store metadata file."""
        return self._dir / "__meta__.json"

    @property
    def standard_csv(self) -> Path:
        """ The Store standardization file."""
        return self._dir / "__standard__.csv" if self.is_standardized else Path()

    @property
    def data(self) -> Frame:
        """ The Store data."""
        self._data = Frame(self.data_csv) if self._data is None else self._data
        return self._data

    @property
    def meta(self) -> dict:
        """ The Store metadata."""
        return self._meta

    @property
    def K(self) -> int:
        """ The number of folds contained in this Store."""
        return self._meta['K']

    @property
    def N(self) -> int:
        """ The number of datapoints (rows of data)."""
        return self._meta['data']['N']

    @property
    def M(self) -> int:
        """ The number of input columns of data."""
        return self._meta['data']['M']

    @property
    def L(self) -> int:
        """ The number of output columns of data."""
        return self._meta['data']['L']

    @property
    def standard(self) -> Frame:
        """ The Store standard: A (2, M+L) DataFrame consisting of (,M+L) ``loc`` values in the first row, (,M+L) ``scale`` values in the second."""
        self._standard = Frame(self.standard_csv) if self._standard is None else self._standard
        return self._standard

    def _read_meta_json(self) -> dict:
        # noinspection PyTypeChecker
        with open(self.meta_json, mode='r') as file:
            return json.load(file)

    def _write_meta_json(self):
        # noinspection PyTypeChecker
        with open(self.meta_json, mode='w') as file:
            json.dump(self._meta, file, indent=8)

    @property
    def is_standardized(self) -> bool:
        return self._meta['standard'] != Store.Standard.none.__name__

    def create_standardized_frame(self, csv: PathLike, df: DataFrame) -> Frame:
        """ Overwrite ``df`` with its standardized version, saving to csv.

        Args:
            df: The data to standardize.
            csv: Locates the return Frame.
        Returns: A Frame written to csv containing df, standardized.
        """
        if self.is_standardized:
            df = (df - self.standard.df.iloc[0]) / self.standard.df.iloc[1]
        return Frame(csv, df)

    def standardize(self, standard: Standard.Specification) -> Frame:
        """ Standardize this Store, and update ``self.meta``. If ``standard==Standard.none``, nothing else is done.
        Otherwise, ``standard`` (a function member of Standard.Specification) is applied to ``self.data.df``, standardizing it,
        and writing the files ``[self.standard_csv]`` and ``[self.data_csv]``.

        Args:
            standard: Specification of Standard, either Standard.none, Standard.mean_and_range or Standard.mean_and_std.
        Returns: self.standard.
        """
        self._meta['standard'] = standard.__name__
        self._write_meta_json()
        self._standard = Frame(self.standard_csv, standard(self.data.df))
        if self.is_standardized:
            self._data = self.create_standardized_frame(self.data_csv, self._data.df)
        return self._standard

    def fold_dir(self, k: int) -> Path:
        """ Returns the path containing each fold between 0 and K.

        Args:
            k: The fold which the function is creating the path for.

        Raises:
            AssertionError: if k is not in range(K).
        """
        assert 0 <= k < self.K, "Requested fold {k:d} of a {K:d}-fold data.Store".format(k=k, K=self.K)
        return self.dir / "fold.{k:d}".format(k=k)

    def _K_folds_update(self, K: int, shuffled_before_folding: bool):
        self._meta.update({'K': K, 'shuffled before folding': shuffled_before_folding})
        self._write_meta_json()

    # noinspection PyUnresolvedReferences,PyUnresolvedReferences
    def split(self):
        """ Split this Store into L Splits by output. Each Split l is just a Store (whose L=1) containing the lth output only."""
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
            if self.is_standardized:
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
    def splits(self) -> List[Tuple[int, Path]]:
        """ Lists the index and path of every Split in this Store."""
        return [(int(split_dir.suffix[1:]), split_dir) for split_dir in self.dir.glob("split.[0-9]*")]

    def meta_data_update(self):
        """ Update __meta__"""
        self._meta.update({'data': {'X_heading': self._data.df.columns.values[0][0],
                                    'Y_heading': self._data.df.columns.values[-1][0]}})
        self._meta['data'].update({'N': self.data.shape[0], 'M': self.X.shape[1],
                                   'L': self.Y.shape[1]})
        self._write_meta_json()

    @property
    def X(self) -> DataFrame:
        """ The input X, as an (N,M) design Matrix with column headings."""
        return self.data.df[self._meta['data']['X_heading']]

    @property
    def Y(self) -> DataFrame:
        """ The output Y as an (N,L) Matrix with column headings."""
        return self.data.df[self._meta['data']['Y_heading']]

    def __init__(self, dir_: PathLike, init_mode: InitMode = InitMode.READ):
        """ Initialize Store.

        Args:
            dir_: The location (directory) of the Store.
            init_mode: The mode to initialize with, variations on READ (an existing Store) and CREATE (a new one).
        """
        self._dir = Path(dir_)
        self._data = None
        self._standard = None
        if init_mode <= Store.InitMode.READ:
            self._meta = self._read_meta_json()
            if init_mode is Store.InitMode.READ:
                self._data = Frame(self.data_csv)
        else:
            self._dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    @classmethod
    def from_df(cls, dir_: PathLike, df: DataFrame, meta: Dict = DEFAULT_META) -> 'Store':
        """ Create a Store from a DataFrame.

        Args:
            dir_: The location (directory) of the Store.
            df: The data to store in [Return].data_csv.
            meta: The metadata to store in [Return].meta_json.
        Returns: A new Store.
        """
        store = Store(dir_, Store.InitMode.CREATE)
        store._meta = {**cls.DEFAULT_META, **meta}
        store._data = Frame(store.data_csv, df)
        store.meta_data_update()
        store.standardize(Store.Standard.none)
        return store

    DEFAULT_ORIGIN_CSV_KWARGS = {'skiprows': None, 'index_col': None}

    @classmethod
    def from_csv(cls, dir_: PathLike, csv: PathLike, meta: Dict = DEFAULT_META, skiprows: ZeroOrMoreInts = None, **kwargs) -> 'Store':
        """ Create a Store from a csv file.

        Args:
            dir_: The location (directory) of the Store.
            csv: File containing the data to store in [Return].data_csv.
            meta: The metadata to store in [Return].meta_json.
            skiprows: The rows of csv to skip while reading, a convenience update to csv_kwargs.
        Keyword Args:
            kwargs: Updates Store.DEFAULT_ORIGIN_CSV_KWARGS for csv reading, as detailed in
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html.
        Returns: A new Store.
        """
        csv = Path(csv)
        _meta = {**Store.DEFAULT_META, **meta}
        origin_csv_kwargs = {**cls.DEFAULT_ORIGIN_CSV_KWARGS, **kwargs, **{'skiprows': skiprows}}
        data = Frame(csv, **origin_csv_kwargs)
        _meta['origin'] = {'csv': str(csv.absolute()), 'origin_csv_kwargs': origin_csv_kwargs}
        return cls.from_df(dir_, data.df, _meta)


class Fold(Store):
    """ A Fold is defined as a dir containing a ``__data__.csv``, a ``__meta__.json`` file and a ``__test__.csv`` file.
    A Fold is a Store equipped with a test DataFrame backed by ``__test__.csv``.
    """

    def set_test_data(self, df: DataFrame):
        """ Set the test data for this Fold.

        args:
            **df**: The test data to be used.
        """
        self._test = self.create_standardized_frame(self.test_csv, df)

    @property
    def test_csv(self) -> Path:
        """ The test data file. Must be identical in format to the self.data_csv file."""
        return self._dir / "__test__.csv"

    @property
    def test(self) -> Frame:
        """ The test data."""
        return self._test

    @property
    def test_Y(self) -> DataFrame:
        """ The test output Y as an (N,L) Matrix with column headings."""
        return self._test.df[self._meta['data']['Y_heading']]

    @property
    def test_X(self) -> DataFrame:
        """ The test input X, as an (NTest,M) design Matrix with column headings."""
        return self._test.df[self._meta['data']['X_heading']].iloc[:, :self._M]

    @property
    def M(self) -> int:
        """ The number of input columns of data."""
        return self._M

    @property
    def X(self) -> DataFrame:
        """ The input X, as an (N,M) design Matrix with column headings."""
        return super().X.iloc[:, 0:self._M]

    def __init__(self, parent: Union[Store, PathLike], k: int, M: int = -1):
        """ Initialize Fold by reading existing files. Creation is handled by the classmethod Fold.into_K_folds.

        Args:
            parent: The parent Store, or its dir.
            k: The index of the Fold within parent.
            M: The number of input columns used. If not 0 &lt M &lt self.M, all columns are used.

        Raises:
            AssertionError: Unless 0 &lt= k &lt parent.K
        """
        if not isinstance(parent, Store):
            parent = Store(parent, Store.InitMode.READ_META_ONLY)
        assert 0 <= k < parent.K, \
            "Fold k={k:d} is out of bounds 0 <= k < K = {K:d} in data.Store({parent_dir:s}".format(k=k, K=parent.K, parent_dir=str(parent.dir))
        super().__init__(parent.fold_dir(k))
        self._M = M if 0 < M < super().M else super().M
        self._test = Frame(self.test_csv)

    DEFAULT_META = {'parent_dir': "", 'k': -1, 'K': -1}

    @classmethod
    def into_K_folds(cls, parent: Store, K: int, shuffled_before_folding: bool = True,
                     standard: Store.Standard.Specification = Store.Standard.mean_and_std, replace_empty_test_with_data_: bool = True):
        """ Fold parent into K Folds for testing.

        Args:
            parent: The Store to fold into K.
            K: The number of Folds, between 1 and N inclusive.
            shuffled_before_folding: Whether to shuffle the samples before sampling.
            If False, each Fold.test will contain 1 sample from the first K samples in parent.__data__, 1 sample from the second K samples, and so on.
            standard: Specification of Standard, either Standard.none, Standard.mean_and_range or Standard.mean_and_std.
            replace_empty_test_with_data_: Whether to replace an empty test file with the training data when K==1.

        Raises:
            AssertionError: Unless 1 &lt= K &lt= N.
        """
        N = len(parent.data.df.index)
        assert 1 <= K <= N, "K={K:d} does not lie between 1 and N=len(self.data.df.index)={N:d} inclusive".format(K=K, N=N)

        for k in range(K, parent.K):
            parent.fold_dir(k).mkdir(mode=0o777, parents=False, exist_ok=True)
            parent.fold_dir(k).rmdir()
        parent._K_folds_update(K, shuffled_before_folding)

        indices = list(range(N))
        if shuffled_before_folding:
            shuffle(indices)

        # noinspection PyUnresolvedReferences,PyUnresolvedReferences,PyUnresolvedReferences
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
        """ A function that renames any directories called fold_# to fold.#.
        This method is only included for backward compatibility, and is not intended for general use.
        """
        for p in dir_.iterdir():
            if p.is_dir():
                Fold._rename(p)
                split_name = p.name.split("_")
                if split_name[0] == "fold":
                    p.rename(p.parent / (split_name[0] + "." + split_name[1]))
