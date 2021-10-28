#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2021 Robert A. Milton. All rights reserved.
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

""" Contains data storage structures."""

from __future__ import annotations

from romcomma.typing_ import *
from romcomma.test import distributions
from copy import deepcopy
from itertools import chain
from random import shuffle
from pathlib import Path
from numpy import NaN, append, transpose, sqrt
from pandas import DataFrame, Series, read_csv, concat
from enum import IntEnum, auto
import json


class Frame:
    """ Encapsulates a DataFrame (df) backed by a source file."""
    @classmethod
    @property
    def DEFAULT_CSV_OPTIONS(cls) -> Dict[str, Any]:
        """ The default options (kwargs) to pass to pandas.read_csv."""
        return {'sep': ',', 'header': [0, 1], 'index_col': 0, }

    @property
    def csv(self) -> Path:
        """ The csv file."""
        return self._csv

    @property
    def is_empty(self) -> bool:
        """ Defines the empty Frame as that having an empty Path."""
        return 0 == len(self._csv.parts)

    def write(self):
        """ Write to csv, according to Frame.DEFAULT_CSV_OPTIONS."""
        assert not self.is_empty, 'Cannot write when frame.is_empty.'
        self.df.to_csv(path_or_buf=self._csv, sep=Frame.DEFAULT_CSV_OPTIONS['sep'], index=True)

    # noinspection PyDefaultArgument
    def __init__(self, csv: PathLike = Path(), df: DataFrame = DataFrame(), **kwargs):
        """ Initialize Frame.

        Args:
            csv: The csv file.
            df: The initial data. If this is empty, it is read from csv, otherwise it overwrites (or creates) csv.
        Keyword Args:
            kwargs: Updates Frame.DEFAULT_CSV_OPTIONS for csv reading as detailed in
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html.
                This is not relevant to writing, which just uses Frame.DEFAULT_CSV_OPTIONS.
        """
        self._csv = Path(csv)
        if self.is_empty:
            assert df.empty, 'csv is an empty path, but df is not an empty DataFrame.'
            self.df = df
        elif df.empty:
            self.df = read_csv(self._csv, **{**Frame.DEFAULT_CSV_OPTIONS, **kwargs})
        else:
            self.df = df
            self.write()


class Store:
    """ A ``store`` object is defined as a ``store.folder`` containing a ``store.csv`` file and a ``store.meta_json`` file.

    These files specify the global dataset to be analyzed. This dataset must be further split into Folds contained within the Store.
    """

    class InitMode(IntEnum):
        READ_META_ONLY = auto()
        READ = auto()
        CREATE = auto()

    @property
    def folder(self) -> Path:
        """ The Store folder."""
        return self._folder

    @property
    def csv(self) -> Path:
        """ The Store data file."""
        return self._folder / '__data__.csv'

    @property
    def data(self) -> Frame:
        """ The Store data."""
        self._data = Frame(self.csv) if self._data is None else self._data
        return self._data

    @property
    def X(self) -> DataFrame:
        """ The input X, as an (N,M) design Matrix with column headings."""
        return self.data.df.loc[self._meta['data']['X_heading']]

    @property
    def Y(self) -> DataFrame:
        """ The output Y as an (N,L) Matrix with column headings."""
        return self.data.df.loc[self._meta['data']['Y_heading']]

    @property
    def meta_json(self) -> Path:
        """ The Store metadata file."""
        return self._folder / '__meta__.json'

    @property
    def meta(self) -> dict:
        """ The Store metadata."""
        return self._meta

    def _read_meta_json(self) -> dict:
        with open(self.meta_json, mode='r') as file:
            return json.load(file)

    def _write_meta_json(self):
        with open(self.meta_json, mode='w') as file:
            json.dump(self._meta, file, indent=8)

    def meta_update(self):
        """ Update __meta__"""
        self._meta.update({'data': {'X_heading': self._data.df.columns.values[0][0],
                                    'Y_heading': self._data.df.columns.values[-1][0]}})
        self._meta['data'].update({'N': self.data.df.shape[0], 'M': self.X.shape[1],
                                   'L': self.Y.shape[1]})
        self._write_meta_json()

    @property
    def N(self) -> int:
        """ The number of datapoints (rows of data)."""
        return self._meta['data']['N']

    @property
    def M(self) -> int:
        """ The number of input columns in `self.data`."""
        return self._meta['data']['M']

    @property
    def L(self) -> int:
        """ The number of output columns in `self.data`."""
        return self._meta['data']['L']

    @property
    def K(self) -> int:
        """ The number of folds contained in this Store."""
        return self._meta['K']

    def into_K_folds(self, K: int, shuffle_before_folding: bool = True):
        """ Fold parent into K Folds for testing.

        Args:
            parent: The Store to fold into K.
            K: The number of Folds, between 1 and N inclusive.
            shuffled_before_folding: Whether to shuffle the samples before sampling.
            If False, each Fold.test_data will contain 1 sample from the first K samples in parent.__data__, 1 sample from the second K samples, and so on.
            standard: Specification of Standard, either Standard.none, Standard.mean_and_range or Standard.mean_and_std.
            replace_empty_test_with_data_: Whether to replace an empty test_data file with the training data when K==1.

        Raises:
            ValueError: Unless 1 &lt= K &lt= N.
        """
        N = len(self.data.df.index)
        if not (1 <= K <= N):
            raise ValueError(f'K={K:d} does not lie between 1 and N={N:d} inclusive.')
        for k in range(K, self.K):
            self.fold_folder(k).mkdir(mode=0o777, parents=False, exist_ok=True)
            self.fold_folder(k).rmdir()
        self._meta.update({'K': K, 'shuffle before folding': shuffled_before_folding})
        self._write_meta_json()
        indices = list(range(N))
        if shuffled_before_folding:
            shuffle(indices)

        # noinspection PyUnresolvedReferences
        def __fold_from_indices(_k: int, train: List[int], test: List[int]):
            assert len(train) > 0
            meta = {**Fold.DEFAULT_META, **{'parent_dir': str(parent.folder), 'k': _k, 'K': parent.K}}
            fold = Store.from_df(parent.fold_folder(_k), parent.data.df.iloc[train], meta)
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

        def __indicators():
            # noinspection PyUnusedLocal
            K_blocks = [list(range(K)) for i in range(int(N / K))]
            K_blocks.append(list(range(N % K)))
            for K_range in K_blocks:
                shuffle(K_range)
            return list(chain(*K_blocks))

        if 1 == K:
            __fold_from_indices(_k=0, train=indices, test=[])
        else:
            indicators = __indicators()
            for k in range(K):
                __fold_from_indices(_k=k,
                                    train=[index for index, indicator in zip(indices, indicators) if k != indicator],
                                    test=[index for index, indicator in zip(indices, indicators) if k == indicator])
        return K

    def fold_folder(self, k: int) -> Path:
        """ Returns the path containing each fold between 0 and K.

        Args:
            k: The fold which the function is creating the path for.
        """
        return self.folder / f'fold.{k:d}'

    def split(self):
        """ Split this Store into L Splits by output. Each Split l is just a Store (whose L=1) containing the lth output only."""
        for l in range(self.L):
            destination = ((self.folder.parent / f'split.{l:d}') / self.folder.name if self.__class__ == Fold
                           else self.folder / f'split.{l:d}')
            if not destination.exists():
                destination.mkdir(mode=0o777, parents=True, exist_ok=False)
            indices = append(range(self.M), self.M + l)
            data = self.data.df.take(indices, axis=1, is_copy=True)
            Frame(destination / self.csv.name, data)
            meta = deepcopy(self._meta)
            meta['data']['L'] = 1
            with open(destination / self.meta_json.name, mode='w') as file:
                json.dump(meta, file, indent=8)
            if self.is_standardized:
                standard = self.standard.df.take(indices, axis=1, is_copy=True)
                Frame(destination / self.standard_csv.name, standard)
            if self.__class__ == Fold:
                # noinspection PyUnresolvedReferences
                test = self._test.df.take(indices, axis=1, is_copy=True)
                # noinspection PyUnresolvedReferences
                Frame(destination / self.test_csv.name, test)
            else:
                for k in range(self.K):
                    fold = Fold(self, k)
                    fold.split()

    @property
    def splits(self) -> List[Tuple[int, Path]]:
        """ Lists the index and path of every Split in this Store."""
        return [(int(split_dir.suffix[1:]), split_dir) for split_dir in self.folder.glob('split.[0-9]*')]

    def __init__(self, folder: PathLike, init_mode: InitMode = InitMode.READ):
        """ Initialize Store.

        Args:
            folder: The location (folder) of the Store.
            init_mode: The mode to initialize with, variations on READ (an existing Store) and CREATE (a new one).
        """
        self._folder = Path(folder)
        self._data = None
        if init_mode <= Store.InitMode.READ:
            self._meta = self._read_meta_json()
            if init_mode is Store.InitMode.READ:
                self._data = Frame(self.csv)
        else:
            self._folder.mkdir(mode=0o777, parents=True, exist_ok=True)

    @classmethod
    @property
    def DEFAULT_META(cls) -> Dict[str, Any]:
        """ Default meta data for a store."""
        return {'csv_kwargs': Frame.DEFAULT_CSV_OPTIONS, 'data': {}, 'K': 0, 'shuffle before folding': False}

    @classmethod
    @property
    def DEFAULT_CSV_OPTIONS(cls) -> Dict[str, Any]:
        """ The default options (kwargs) to pass to pandas.read_csv."""
        return {'skiprows': None, 'index_col': None}

    @classmethod
    def from_df(cls, folder: PathLike, df: DataFrame, meta: Dict = DEFAULT_META) -> Store:
        """ Create a Store from a DataFrame.

        Args:
            folder: The location (folder) of the Store.
            df: The data to store in [Return].csv.
            meta: The meta data to store in [Return].meta_json.
        Returns: A new Store.
        """
        store = Store(folder, Store.InitMode.CREATE)
        store._meta = {**cls.DEFAULT_META, **meta}
        store._data = Frame(store.csv, df)
        store.meta_update()
        return store

    @classmethod
    def from_csv(cls, folder: PathLike, csv: PathLike, meta: Dict = DEFAULT_META, skiprows: ZeroOrMoreInts = None, **kwargs) -> Store:
        """ Create a Store from a csv file.

        Args:
            folder: The location (folder) of the target Store.
            csv: The file containing the data to store in [Return].csv.
            meta: The meta data to store in [Return].meta_json.
            skiprows: The rows of csv to skip while reading, a convenience update to csv_kwargs.
        Keyword Args:
            kwargs: Updates Store.DEFAULT_CSV_OPTIONS for reading the csv file, as detailed in
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html.
        Returns: A new Store located in folder.
        """
        csv = Path(csv)
        origin_csv_kwargs = {**cls.DEFAULT_CSV_OPTIONS, **kwargs, **{'skiprows': skiprows}}
        data = Frame(csv, **origin_csv_kwargs)
        meta['origin'] = {'csv': str(csv.absolute()), 'origin_csv_kwargs': origin_csv_kwargs}
        return cls.from_df(folder, data.df, meta)


class Fold(Store):
    """ A Fold is defined as a folder containing a ``__data__.csv``, a ``__meta__.json`` file and a ``__test__.csv`` file.
    A Fold is a Store equipped with a test_data DataFrame backed by ``__test__.csv``.

    Additionally, a fold can reduce the dimensionality ``M`` of the input ``X``.
    """

    @property
    def normalization(self) -> Normalization:
        return self._normalization

    @property
    def test_csv(self) -> Path:
        """ The test_data data file. Must be identical in format to the self.csv file."""
        return self.folder / '__test__.csv'

    @property
    def test_data(self) -> Frame:
        """ The test_data data."""
        return self._test_data

    @property
    def test_y(self) -> DataFrame:
        """ The test_data output y as an (n,L) Matrix with column headings."""
        return self._test_data.df[self._meta['data']['Y_heading']]

    @property
    def test_x(self) -> DataFrame:
        """ The test_data input x, as an (n,M) design Matrix with column headings."""
        return self._test_data.df[self._meta['data']['X_heading']]

    def __init__(self, parent: Union[Store, PathLike], k: int):
        """ Initialize Fold by reading existing files. Creation is handled by the classmethod Fold.from_dfs.

        Args:
            parent: The parent Store, or its folder.
            k: The index of the Fold within parent.
            M: The number of input columns used. If not 0 &lt M &lt self.M, all columns are used.
        """
        if not isinstance(parent, Store):
            parent = Store(parent, Store.InitMode.READ_META_ONLY)
        assert 0 <= k < parent.K, f'Fold k={k:d} is out of bounds 0 <= k < K = {self.K:d} in data.Store({parent.folder:s}'
        super().__init__(parent.fold_folder(k))
        self._test_data = Frame(self.test_csv)
        self._normalization = Normalization(self)

    @classmethod
    def from_dfs(cls, parent: Store, k: int, data: DataFrame, test_data: DataFrame) -> Fold:
        """ Create a Fold from a DataFrame.

        Args:
            folder: The location (folder) of the Store.
            df: The data to store in [Return].csv.
            meta: The meta data to store in [Return].meta_json.
        Returns: A new Store.
        """
        fold = Fold(parent, k)
        fold._meta = {**parent.meta, **{'k': k}}
        fold._normalization = Normalization(fold, data)
        fold._data = Frame(fold.csv, fold.normalization.apply_to(data))
        fold._test_data = Frame(fold.test_csv, fold.normalization.apply_to(test_data))
        fold.meta_update()
        return fold


class Normalization:
    """ Encapsulates Specifications for standardizing data as ``classmethods``.

    A Specification is a function taking an (N,M+L) DataFrame ``df `` (to be standardized) as input and returning a (2,M+L) DataFrame.
    The first row of the return contains (,M+L) ``loc`` values, the second row (,M+L) ``scale`` values.

    Ultimately the Store class standardizes ``df`` to ``(df - loc)/scale``.
    """

    @property
    def fold(self) -> Fold:
        return self._fold

    @property
    def csv(self) -> Path:
        """ The normalization file."""
        return self._fold.folder / '__normalization__.csv'

    @property
    def frame(self) -> Frame:
        """ The normalization frame."""
        return Frame(self.csv) if self._frame is None else self._frame

    def apply_to(self, df: DataFrame) -> DataFrame:
        df = (df - self.frame.df.loc['mean']) / self.frame.df.loc['range']
        df.loc[self._fold.meta['X_heading']] = self._standard_normal.ppf(df.loc[self._fold.meta['X_heading']])
        return df

    def undo_from(self, df: DataFrame) -> DataFrame:
        df.loc[self._fold.meta['X_heading']] = self._standard_normal.cdf(df.loc[self._fold.meta['X_heading']])
        df = df * self.frame.df.loc['range'] + self.frame.df.loc['mean']
        return df

    def _create_frame_from(self, data: DataFrame) -> Frame:
        mean = data.mean()
        mean.name = 'mean'
        std = data.std()
        std.name = 'std'
        semi_range = std * sqrt(3)
        semi_range.name = 'rng'
        mmin = mean - semi_range
        mmin.name = 'min'
        mmax = mean + semi_range
        mmax.name = 'max'
        df = concat((mean, std, 2 * semi_range, mmin, mmax), axis=1)
        return Frame(self.csv, df.T)

    def __init__(self, fold: Fold, data: Optional[DataFrame] = None, test_data: Optional[DataFrame] = None):
        assert ((data is None and test_data is None) or (data is not None and test_data is not None),
                'Normalize should be initialized with both data and test_data, or neither.')
        self._fold = fold
        self._standard_normal = distributions.Univariate(name='norm', loc=0, scale=1).parametrized
        if data is None and test_data is None:
            self._frame = Frame(self.csv) if self.csv.exists() else None
        else:
            self._frame = self._create_frame_from(data)
