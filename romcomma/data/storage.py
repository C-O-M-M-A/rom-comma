#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2023 Robert A. Milton. All rights reserved.
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

""" Data structures for storage """

from __future__ import annotations

from romcomma.base.definitions import *
from copy import deepcopy
import itertools
import random
import shutil
from enum import IntEnum, auto
import scipy.stats
import json


class Frame:
    """ Encapsulates a pd.DataFrame (df) backed by a source file."""

    @classmethod
    @property
    def CSV_OPTIONS(cls) -> Dict[str, Any]:
        """ The default options (kwargs) to pass to pandas.pd.read_csv."""
        return {'sep': ',', 'header': [0, 1], 'index_col': 0, }

    @property
    def csv(self) -> Path:
        return self._csv

    @property
    def is_empty(self) -> bool:
        """ Defines the empty Frame as that having an empty Path."""
        return 0 == len(self._csv.parts)

    def write(self):
        """ Write to csv, according to Frame.CSV_OPTIONS."""
        assert not self.is_empty, 'Cannot write when frame.is_empty.'
        self.df.to_csv(path_or_buf=self._csv, sep=Frame.CSV_OPTIONS['sep'], index=True)

    def __repr__(self) -> str:
        return str(self._csv)

    def __str__(self) -> str:
        return self._csv.name

    # noinspection PyDefaultArgument
    def __init__(self, csv: Path | str = Path(), df: pd.DataFrame = pd.DataFrame(), **kwargs):
        """ Initialize Frame.

        Args:
            csv: The csv file.
            df: The initial data. If this is empty, it is read from csv, otherwise it overwrites (or creates) csv.
        Keyword Args:
            kwargs: Updates Frame.CSV_OPTIONS for csv reading as detailed in
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html.
                This is not relevant to writing, which just uses Frame.CSV_OPTIONS.
        """
        self._csv = Path(csv)
        if self.is_empty:
            assert df.empty, 'csv is an empty path, but df is not an empty pd.DataFrame.'
            self.df = df
        elif df.empty:
            self.df = pd.read_csv(self._csv, **{**Frame.CSV_OPTIONS, **kwargs})
        else:
            self.df = df
            self.write()


class Repository:
    """ A ``repo`` object is defined as a folder containing a ``data.csv`` file and a ``meta.json`` file.

    These files specify the global dataset to be analyzed. This dataset must be further split into Folds contained within the Repository.
    """

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def data(self) -> Frame:
        return self._data

    @property
    def X(self) -> pd.DataFrame:
        """ The input X, as an (N,M) design Matrix with column headings."""
        return self._data.df[self._meta['data']['X_heading']]

    @property
    def Y(self) -> pd.DataFrame:
        """ The output Y as an (N,L) Matrix with column headings."""
        return self._data.df[self._meta['data']['Y_heading']]

    def read_meta(self) -> Dict[str, Any]:
        with open(self._meta_json, mode='r') as file:
            return json.load(file)

    def write_meta(self):
        with open(self._meta_json, mode='w') as file:
            json.dump(self._meta, file, indent=8)

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta

    def _update_meta(self):
        self._meta.update({'data': {'X_heading': self._data.df.columns.values[0][0],
                                    'Y_heading': self._data.df.columns.values[-1][0]}})
        self._meta['data'].update({'N': self.data.df.shape[0], 'M': self.X.shape[1],
                                   'L': self.Y.shape[1]})
        self.write_meta()

    @property
    def N(self) -> int:
        """ The number of samples (rows of data)."""
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
        """ The number of folds contained in this Repository."""
        return self._meta['K']

    def clean_copy(self, dst: Path | str):
        """ Make a clean copy of this repo.

        Args:
            dst: The location of the copy.
        """
    @property
    def folds(self) -> range:
        """ The indices of the folds contained in this Repository."""
        if isinstance(self, Fold) or self.K < 1:
            return range(0, 0)
        else:
            return range(self.K + (1 if self.meta['has_improper_fold'] else 0))

    def into_K_folds(self, K: int, shuffle_before_folding: bool = False, normalization: Optional[Path | str] = None) -> Repository:
        """ Fold this repo into K Folds, indexed by range(K).

        Args:
            K: The number of Folds, of absolute value between 1 and N inclusive.
                An improper Fold, indexed by K and including all data for both training and testing is included by default.
                To suppress this give K as a negative integer.
            shuffle_before_folding: Whether to shuffle the data before sampling.
            normalization: An optional normalization.csv file to use.
        Returns: ``self``, for chaining calls.
        Raises:
            IndexError: Unless 1 &lt= K &lt= N.
        """
        data = self.data.df
        N = data.shape[0]
        if not (1 <= abs(K) <= N):
            raise IndexError(f'K={K:d} does not lie between 1 and N={N:d} inclusive.')
        for k in range(max(abs(K), self.K) + 1):
            shutil.rmtree(self.fold_folder(k), ignore_errors=True)
        index = list(range(N))
        if shuffle_before_folding:
            random.shuffle(index)
        self._meta.update({'K': abs(K), 'shuffle before folding': shuffle_before_folding, 'has_improper_fold': K > 0})
        self.write_meta()
        normalization = Normalization(self, self._data.df).csv if normalization is None else normalization
        if K > 0:
            Fold.from_dfs(parent=self, k=K, data=data.iloc[index], test_data=data.iloc[index], normalization=normalization)
        K = abs(K)
        K_blocks = [list(range(K)) for dummy in range(int(N / K))]
        K_blocks.append(list(range(N % K)))
        for K_range in K_blocks:
            random.shuffle(K_range)
        indicator = list(itertools.chain(*K_blocks))
        for k in range(K):
            indicated = tuple(zip(index, indicator))
            data_index = [index for index, indicator in indicated if k != indicator]
            test_index = [index for index, indicator in indicated if k == indicator]
            data_index = test_index if data_index == [] else data_index
            Fold.from_dfs(parent=self, k=k, data=data.iloc[data_index], test_data=data.iloc[test_index], normalization=normalization)
        return self

    def rotate_folds(self, rotation: NP.Matrix | None) -> Repository:
        """ Uniformly rotate the Folds in a Repository. The rotation (like normalization) applies to each fold, not the repo itself.

        Args:
            rotation: The (M,M) rotation matrix to apply to the inputs. If None, the identity matrix is used.
            If the matrix supplied has the wrong dimensions or is not orthogonal, a random rotation is generated and used instead.
        Returns: ``self``, for chaining calls.
        """
        M = self.M
        if rotation is None:
            rotation = np.eye(M)
        elif rotation.shape != (M, M) or not np.allclose(np.dot(rotation, rotation.T), np.eye(M)):
            rotation = scipy.stats.special_ortho_group.rvs(M)
        for k in self.folds:
            Fold(self, k).X_rotation = rotation
        return self

    def fold_folder(self, k: int) -> Path:
        return self._folder / f'fold.{k:d}'

    def Y_split(self):
        """Split this Repository into L Y_splits. Each Y.l is just a Repository containing the lth output only.

        Raises:
            TypeError: if self is a Fold.
        """
        if isinstance(self, Fold):
            raise TypeError('Cannot Y_split a Fold, only a Repository.')
        for l in range(self.L):
            destination = self.folder / f'Y.{l:d}'
            if not destination.exists():
                destination.mkdir(mode=0o777, parents=True, exist_ok=False)
            indices = np.append(range(self.M), self.M + l)
            data = self.data.df.take(indices, axis=1, is_copy=True)
            Frame(destination / self._csv.name, data)
            meta = deepcopy(self._meta)
            meta['data']['L'] = 1
            Repository.from_df(destination, data, meta)

    @property
    def Y_splits(self) -> List[Tuple[int, Path]]:
        """ Lists the index and path of every Y_split in this Repository."""
        return [(int(Y_dir.suffix[1:]), Y_dir) for Y_dir in self.folder.glob('Y.[0-9]*')]

    # noinspection PyArgumentList
    class _InitMode(IntEnum):
        READ_META_ONLY = auto()
        READ = auto()
        CREATE = auto()

    def __repr__(self) -> str:
        return str(self._folder)

    def __str__(self) -> str:
        return self._folder.name

    def __init__(self, folder: Path | str, **kwargs):
        self._folder = Path(folder)
        self._meta_json = self._folder / 'meta.json'
        self._csv = self._folder / 'data.csv'
        self._data = None
        init_mode = kwargs.get('init_mode', Repository._InitMode.READ)
        if init_mode <= Repository._InitMode.READ:
            self._meta = self.read_meta()
            if init_mode is Repository._InitMode.READ:
                self._data = Frame(self._csv)
        else:
            shutil.rmtree(self._folder, ignore_errors=True)
            self._folder.mkdir(mode=0o777, parents=True, exist_ok=False)

    @classmethod
    @property
    def META(cls) -> Dict[str, Any]:
        return {'csv_kwargs': Frame.CSV_OPTIONS, 'data': {}, 'K': 0, 'shuffle before folding': False}

    @classmethod
    def from_df(cls, folder: Path | str, df: pd.DataFrame, meta: Dict | None = None) -> Repository:
        """ Create a Repository from a pd.DataFrame.

        Args:
            folder: The location (folder) of the Repository.
            df: The data to record in [Return].csv.
            meta: The metadata to record in [Return].meta.json.
        Returns: A new Repository.
        """
        repo = Repository(folder, init_mode=Repository._InitMode.CREATE)
        repo._meta = cls.META | ({} if meta is None else meta)
        repo._data = Frame(repo._csv, df)
        repo._update_meta()
        return repo

    @classmethod
    @property
    def CSV_OPTIONS(cls) -> Dict[str, Any]:
        return {'skiprows': None, 'index_col': 0}

    @classmethod
    def from_csv(cls, folder: Path | str, csv: Path | str, meta: Dict = None, **kwargs) -> Repository:
        """ Create a Repository from a csv file.

        Args:
            folder: The location (folder) of the target Repository.
            csv: The file containing the data to record in [Return].csv.
            meta: The metadata to record in [Return].meta.json.
            kwargs: Updates Repository.CSV_OPTIONS for reading the csv file, as detailed in
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.pd.read_csv.html.
        Returns: A new Repository located in folder.
        """
        csv = Path(csv)
        origin_csv_kwargs = cls.CSV_OPTIONS | kwargs
        data = Frame(csv, **origin_csv_kwargs)
        meta = cls.META if meta is None else cls.META | meta
        meta['origin'] = {'csv': str(csv.absolute()), 'origin_csv_kwargs': origin_csv_kwargs}
        return cls.from_df(folder, data.df, meta)


class Fold(Repository):
    """ A Fold is defined as a folder containing a ``data.csv``, a ``meta.json`` file and a ``test.csv`` file.
    A Fold is a Repository equipped with a test_data pd.DataFrame backed by ``test.csv``.

    Additionally, a fold can reduce the dimensionality ``M`` of the input ``X``.
    """

    @property
    def normalization(self) -> Normalization:
        return self._normalization

    @property
    def test_csv(self) -> Path:
        return self._test_csv

    @property
    def test_data(self) -> Frame:
        return self._test_data

    @property
    def test_x(self) -> pd.DataFrame:
        """ The test_data input x, as an (n,M) design Matrix with column headings."""
        return self._test_data.df[self._meta['data']['X_heading']]

    @property
    def test_y(self) -> pd.DataFrame:
        """ The test_data output y as an (n,L) Matrix with column headings."""
        return self._test_data.df[self._meta['data']['Y_heading']]

    def _X_rotate(self, frame: Frame, rotation: NP.Matrix):
        """ Rotate the input variables in a Frame.

        Args:
            frame: The frame to rotate. Will be written after rotation.
            rotation: The rotation Matrix.
        """
        frame.df.iloc[:, :self.M] = np.einsum('Nm,Mm->NM', frame.df.iloc[:, :self.M], rotation)
        frame.write()

    @property
    def X_rotation(self) -> NP.Matrix:
        """ The rotation matrix applied to the input variables self.X, stored in X_rotation.csv. Rotations are applied and stored cumulatively."""
        return Frame(self._X_rotation, header=[0]).df.values if self._X_rotation.exists() else np.eye(self.M)

    @X_rotation.setter
    def X_rotation(self, value: NP.Matrix):
        """ The rotation matrix applied to the input variables self.X, stored in X_rotation.csv. Rotations are applied and stored cumulatively."""
        self._X_rotate(self._data, value)
        self._X_rotate(self._test_data, value)
        old_value = self.X_rotation
        Frame(self._X_rotation, pd.DataFrame(np.matmul(old_value, value)))

    def __init__(self, parent: Repository, k: int, **kwargs):
        """ Initialize Fold by reading existing files. Creation is handled by the classmethod Fold.from_dfs.

        Args:
            parent: The parent Repository.
            k: The index of the Fold within parent.
            M: The number of input columns used. If not 0 &lt M &lt self.M, all columns are used.
        """
        init_mode = kwargs.get('init_mode', Repository._InitMode.READ)
        super().__init__(parent.fold_folder(k), init_mode=init_mode)
        self._X_rotation = self.folder / 'X_rotation.csv'
        self._test_csv = self.folder / 'test.csv'
        if init_mode == Repository._InitMode.READ:
            self._test_data = Frame(self._test_csv)
            self._normalization = Normalization(self)

    @classmethod
    def from_dfs(cls, parent: Repository, k: int, data: pd.DataFrame, test_data: pd.DataFrame,
                 normalization: Optional[Path | str] = None) -> Fold:
        """ Create a Fold from a pd.DataFrame.

        Args:
            parent: The parent Repository.
            k: The index of the fold to be created.
            data: Training data.
            test_data: Test data.
            normalization: An optional normalization.csv file to use.

        Returns: The Fold created.
        """

        fold = cls(parent, k, init_mode=Repository._InitMode.CREATE)
        fold._meta = cls.META | parent.meta | {'k': k}
        if normalization is None:
            fold._normalization = Normalization(fold, data)
        else:
            fold._normalization = Normalization(fold)
            shutil.copy(Path(normalization), fold._normalization.csv)
        fold._data = Frame(fold._csv, fold.normalization.apply_to(data))
        fold._test_data = Frame(fold._test_csv, fold.normalization.apply_to(test_data))
        fold._update_meta()
        return fold


class Normalization:
    """ Encapsulates the normalization of data.
        X data is assumed to follow a Uniform distribution, which is normalized to U[0,1] , then inverse probability transformed to N[0,1].
        Y data is normalized to zero mean and unit variance.
    """

    @classmethod
    @property
    def UNIFORM_MARGIN(cls) -> float:
        return 1.0E-12

    @property
    def csv(self) -> Path:
        return self._fold.folder / 'normalization.csv'

    @property
    def frame(self) -> Frame:
        self._frame = Frame(self.csv) if self._frame is None else self._frame
        return self._frame

    @property
    def _relevant_stats(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        return (self.frame.df.iloc[self.frame.df.index.get_loc('min'), :self._fold.M], self.frame.df.iloc[self.frame.df.index.get_loc('rng'), :self._fold.M],
                self.frame.df.iloc[self.frame.df.index.get_loc('mean'), self._fold.M:], self.frame.df.iloc[self.frame.df.index.get_loc('std'), self._fold.M:])

    def apply_to(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Apply this normalization.

        Args:
            df: The pd.DataFrame to Normalize.

        Returns: df, Normalized.
        """
        X_min, X_rng, Y_mean, Y_std = self._relevant_stats
        X = df.iloc[:, :self._fold.M].copy(deep=True)
        Y = df.iloc[:, self._fold.M:].copy(deep=True)
        X = X.sub(X_min, axis=1)[X_min.axes[0]].div(X_rng, axis=1)[X_rng.axes[0]].clip(lower=self.UNIFORM_MARGIN, upper=1 - self.UNIFORM_MARGIN)
        X.iloc[:, :] = scipy.stats.norm.ppf(X, loc=0, scale=1)
        Y = Y.sub(Y_mean, axis=1).div(Y_std, axis=1)
        return pd.concat((X, Y), axis=1)

    def undo_from(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Undo this normalization.

        Args:
            df: The (Normalized) pd.DataFrame to UnNormalize.

        Returns: df, UnNormalized.
        """
        X_min, X_rng, Y_mean, Y_std = self._relevant_stats
        X = df.iloc[:, :self._fold.M].copy(deep=True)
        Y = df.iloc[:, self._fold.M:].copy(deep=True)
        X.iloc[:, :] = scipy.stats.norm.cdf(X, loc=0, scale=1)
        X = X.mul(X_rng, axis=1)[X_rng.axes[0]].add(X_min, axis=1)[X_min.axes[0]]
        Y = Y.mul(Y_std, axis=1)[Y_std.axes[0]].add(Y_mean, axis=1)[Y_mean.axes[0]]
        return pd.concat((X, Y), axis=1)

    def unscale_Y(self, dfY: pd.DataFrame) -> pd.DataFrame:
        """ Undo the Y-scaling of this normalization, without adding the Y-Mean. Suitable treatment for unNormalizing SD, for example.

        Args:
            dfY: The (Normalized) pd.DataFrame to UnNormalize.

        Returns: dfY, UnNormalized.
        """
        X_min, X_rng, Y_mean, Y_std = self._relevant_stats
        return dfY.copy(deep=True).mul(Y_std, axis=1)[Y_std.axes[0]]

    def __repr__(self) -> str:
        return str(self.csv)

    def __str__(self) -> str:
        return self.csv.name

    def __init__(self, fold: Repository, data: Optional[pd.DataFrame] = None):
        """ Initialize this Normalization. If the fold has already been Normalized, that Normalization is returned.

        Args:
            fold: The fold to Normalize.
            data: The data from which to calculate Normalization.
        """
        self._fold = fold
        if self.csv.exists():
            self._frame = Frame(self.csv)
        elif data is None:
            self._frame = None
        else:
            mean = data.mean()
            mean.name = 'mean'
            std = data.std()
            std.name = 'std'
            semi_range = std * np.sqrt(3)
            semi_range.name = 'rng'
            m_min = mean - semi_range
            m_min.name = 'min'
            m_max = mean + semi_range
            m_max.name = 'max'
            df = pd.concat((mean, std, 2 * semi_range, m_min, m_max), axis=1)
            self._frame = Frame(self.csv, df.T)
