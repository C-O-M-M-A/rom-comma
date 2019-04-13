""" Encapsulates data storage structures.

Contents:
    :Frame: class.
    :Store: class.
    :Store.Standard: class.
    :Fold(Store): class.
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


class Store:
    """ A Store is defined as a dir containing a ``__data__.csv`` file and a ``__meta__.json`` file.
    These files specify the global dataset to be analyzed.
    """

    class Standard:
        # noinspection SpellCheckingInspection
        """ Encapsulates Specifications for standardizing data as ``classmethods``."""

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
            return DataFrame()

        @classmethod
        def mean_and_range(cls, df: DataFrame) -> DataFrame:
            scale = df.max() - df.min()
            scale.name = 'range'
            result = cls._stack_as_rows(cls._mean(df), scale)
            return result

        @classmethod
        def mean_and_std(cls, df: DataFrame) -> DataFrame:
            scale = df.std()
            scale.name = 'std'
            return cls._stack_as_rows(cls._mean(df), scale)

    META = {'csv_parameters': Frame.CSV_PARAMETERS, 'standard': Standard.none.__name__, 'data': {},
            'K': 0, 'shuffled before folding': False}

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def data_csv(self) -> Path:
        return self._dir / "__data__.csv"

    @property
    def data(self) -> Frame:
        self._data = Frame(self.data_csv) if self._data is None else self._data
        return self._data

    @property
    def meta_json(self) -> Path:
        return self._dir / "__meta__.json"

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def N(self) -> int:
        return self._meta['data']['N']

    @property
    def M(self) -> int:
        return self._meta['data']['M']

    @property
    def L(self) -> int:
        return self._meta['data']['L']

    @property
    def K(self) -> int:
        return self._meta['K']

    def _read_meta_json(self) -> dict:
        # noinspection PyTypeChecker
        with open(self.meta_json, mode='r') as file:
            return json.load(file)

    def _write_meta_json(self):
        # noinspection PyTypeChecker
        with open(self.meta_json, mode='w') as file:
            json.dump(self._meta, file, indent=8)

    @property
    def standard_csv(self) -> Path:
        return self._dir / "__standard__.csv" if self.standardized else Path()

    @property
    def pre_standard_data_csv(self) -> Path:
        return self._dir / "__pre-standard_data__.csv"

    @property
    def standard(self) -> Frame:
        self._standard = Frame(self.standard_csv) if self._standard is None else self._standard
        return self._standard

    @property
    def standardized(self) -> bool:
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
        if 0 <= k < self.K:
            return self.dir / "fold_{k:d}".format(k=k)
        else:
            raise IndexError("Requested fold {k:d} of a {K:d}-fold data.Store".format(k=k, K=self.K))

    # noinspection PyPep8Naming
    def _K_folds_update(self, K: int, shuffled_before_folding: bool):
        self._meta.update({'K': K,
                           'shuffled before folding': shuffled_before_folding})
        self._write_meta_json()

    def _data_update(self):
        self._meta.update({'data': {'X_heading': self._data.df.columns.values[0][0],
                                    'Y_heading': self._data.df.columns.values[-1][0]}})
        self._meta['data'].update({'N': self.data.df.shape[0], 'M': self.X.shape[1],
                                   'L': self.Y.shape[1]})

    # noinspection PyPep8Naming
    @property
    def X(self) -> DataFrame:
        return self.data.df[self._meta['data']['X_heading']]

    # noinspection PyPep8Naming
    @property
    def Y(self) -> DataFrame:
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

    def split(self):
        for l in range(self.L):
            destination = ((self.dir.parent / "split.{0:d}".format(l)) / self.dir.name if self.__class__ == Fold
                           else self.dir / "split.{0:d}".format(l))
            if not destination.exists():
                destination.mkdir(mode=0o777, parents=True, exist_ok=False)
            indices = append(range(self.M), self.M + l)
            data = self.data.df.take(indices, axis=1, is_copy=True)
            data = Frame(destination / self.data_csv.name, data)
            meta = deepcopy(self._meta)
            meta['data']['L'] = 1
            with open(destination / self.meta_json.name, mode='w') as file:
                json.dump(meta, file, indent=8)
            if self.standardized:
                standard = self.standard.df.take(indices, axis=1, is_copy=True)
                standard = Frame(destination / self.standard_csv.name, standard)
            if self.__class__ == Fold:
                test = self._test.df.take(indices, axis=1, is_copy=True)
                test = Frame(destination / self.test_csv.name, test)
            else:
                for k in range(self.K):
                    fold = Fold(self, k)
                    fold.split()

    @property
    def splits(self) -> List[Path]:
        return list(self.dir.glob("split.[0-9]*"))

    class InitMode(IntEnum):
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

    # noinspection PyDefaultArgument
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

    # noinspection PyDefaultArgument
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
        return self._dir / "__test__.csv"

    @property
    def M(self) -> int:
        return self._Xs_taken if self._Xs_taken else super().M

    @property
    def test(self) -> Frame:
        return self._test

    # noinspection PyPep8Naming
    @property
    def test_X(self) -> DataFrame:
        return self._test.df[self._meta['data']['X_heading']] if self._Xs_taken else self._test.df[self._meta['data']['X_heading']]

    # noinspection PyPep8Naming
    @property
    def X(self) -> DataFrame:
        return super().X.iloc[:, 0:self._Xs_taken] if self._Xs_taken else super().X

    # noinspection PyPep8Naming
    @property
    def test_Y(self) -> DataFrame:
        return self._test.df[self._meta['data']['Y_heading']]

    def set_test_data(self, df: DataFrame):
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

    # noinspection PyPep8Naming,PyPep8Naming
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

        def _fold_from_indices(_k: int, train: List[int], test: List[int]):
            assert len(train) > 0
            meta = {**Fold.META, **{'parent_dir': str(parent.dir), 'k_fold': _k,
                                    'K': parent.K}}
            fold = Store.from_df(parent.fold_dir(_k), parent.data.df.iloc[train], meta, overwrite=overwrite)
            fold.standardize(standard, False)
            fold.__class__ = cls
            if len(test) < 1:
                if replace_empty_test_with_data_:
                    # noinspection PyUnresolvedReferences
                    fold._test = fold.create_standardized_frame(fold.test_csv, parent.data.df.iloc[train])
                else:
                    # noinspection PyUnresolvedReferences
                    fold._test = Frame(fold.test_csv,
                                       DataFrame(data=NaN, index=[-1], columns=parent.data.df.columns))
            else:
                # noinspection PyUnresolvedReferences
                fold._test = fold.create_standardized_frame(fold.test_csv, parent.data.df.iloc[test])

        # noinspection PyPep8Naming
        def _indicators():
            # noinspection PyUnusedLocal
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
