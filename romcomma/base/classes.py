#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2024 Robert A. Milton. All rights reserved.
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

""" Base classes for romcomma Models."""

import pandas as pd

from romcomma.base.definitions import *
import shutil
import json
from abc import ABC


class Frame:
    """ Encapsulates a pandas DataFrame backed by a source file."""

    @property
    def csv(self) -> NP.Matrix:
        return self._csv

    @csv.setter
    def csv(self, value: Path | str):
        self._csv = Path(value)
        self.write()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def np(self) -> NP.Matrix:
        return self._df.to_numpy()

    @np.setter
    def np(self, value: NP.Matrix):
        self._df.iloc[:, :] = value
        self.write()

    @property
    def tf(self) -> TF.Matrix:
        return tf.convert_to_tensor(self.np)

    @tf.setter
    def tf(self, value: TF.Matrix):
        self._df.iloc[:, :] = value.numpy()
        self.write()

    def write(self, **kwargs: Any) -> Self:
        """ Write to csv. This is called whenever the data in the Frame changes.

        Args:
            **kwargs: Options passed straight to ``self.to_csv()``.
        Returns: ``self``, for call chaining.
        """
        self._write_options = self._write_options | kwargs
        self._df.to_csv(self._csv, **self._write_options)
        return self

    def broadcast_value(self, target_shape: Tuple[int, int], is_diagonal: bool = True) -> Self:
        """ Broadcast a frame

        Args:
            target_shape: The shape to broadcast to.
            is_diagonal: Whether to zero the off-diagonal elements of a square matrix.
        Returns: Self, for chaining calls.
        Raises:
            IndexError: If broadcasting is impossible.
        """
        try:
            values = np.array(np.broadcast_to(self.np, target_shape))
        except ValueError:
            raise IndexError(f'{repr(self)} has shape {self.df.shape} 'f' which cannot be broadcast to {target_shape}.')
        if is_diagonal and target_shape[0] > 1:
            values = np.diag(np.diagonal(values))
        self._df = pd.DataFrame(values)
        return self.write()

    def __call__(self, *args, **kwargs):
        """ Returns ``self.np``, as this is automatically cast by tf, np and pd."""
        return self.np

    def __repr__(self) -> str:
        return str(self.csv)

    def __str__(self) -> str:
        return self.csv.name

    # noinspection PyDefaultArgument
    def __init__(self, csv: Path | str, data: pd.DataFrame | NP.Array | Iterable | Dict = None, index: pd.Index | NP.ArrayLike = None,
                 columns: pd.Index | NP.ArrayLike = None, dtype: NP.DType | None = None, copy: bool | None = None, **kwargs):
        """ Construct a Frame, from csv or pd.DataFrame. If ``data is None``, the Frame is read from csv. Otherwise the Frame is written to csv.

        Args:
            csv: The csv file path, without ``.csv``.
            data: The data to store. If None, a pd.DataFrame is read from csv. 
                See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            index: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            columns: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            dtype: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            copy: See `pd.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
            **kwargs: Passed straight to `pd.read_csv <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html>`_
                or `DataFrame.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_.
        """
        self._csv = Path(csv)
        self._write_options = {}
        if data is None:
            self._df = (pd.read_csv(self.csv.with_suffix(f'{self.csv.suffix}.csv'), **({'index_col': 0} | kwargs)))
        else:
            self._df = pd.DataFrame(data, index, columns, dtype, copy)
            self.write(**kwargs)


# noinspection PyProtectedMember
class Data(ABC):
    """ Abstraction of Model Data. Essentially a NamedTuple of Frames in a folder.
    Most Data methods are simple wrappers for annoyingly underscored methods of `NamedTuple
    <https://docs.python.org/3/library/collections.html#collections.namedtuple>`_."""

    Matrix: Type = pd.DataFrame | NP.Matrix | TF.Matrix

    class NamedTuple(NamedTuple):
        """ A NamedTuple of data. Must be overridden."""
        NotImplemented: NP.Matrix = np.atleast_2d('NotImplemented')   #: NamedTuple can have any number of members.

    @classmethod
    def make(cls, iterable: Iterable) -> NamedTuple:
        return cls.NamedTuple._make(iterable)

    @classmethod
    @property
    def fields(cls) -> Tuple[str, ...]:
        return cls.NamedTuple._fields

    @classmethod
    @property
    def field_defaults(cls) -> Dict[str, Any]:
        return cls.NamedTuple._field_defaults

    def asdict(self) -> Dict[str, Any]:
        return self._frames._asdict()

    def replace(self, **kwargs: Matrix) -> Self:
        for key, value in kwargs.items():
            value = value.numpy() if isinstance(value, TF.Tensor) else value
            kwargs[key] = value if isinstance(value, Frame) else Frame(self._folder / key, np.atleast_2d(value))
        self._frames = self.NamedTuple(**kwargs) if self._frames is None else self._frames._replace(**kwargs)
        return self

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def frames(self) -> NamedTuple:
        return self._frames

    def move(self, dst_folder: Path | str) -> Self:
        """  Move ``self`` to ``dst_folder``.

        Args:
            dst_folder: The folder to move to. If this exists, it will be emptied.
        Returns: ``self`` for chaining calls.
        """
        self._folder = Data(self.empty(dst_folder), **self.asdict()).folder
        return self

    def __call__(self, *args, **kwargs):
        """ Returns ``self.values``."""
        return self._frames

    def __repr__(self) -> str:
        return str(self._folder)

    def __str__(self) -> str:
        return self._folder.name

    def __init__(self, folder: Path | str, **kwargs: Matrix):
        """ Data Constructor.

        Args:
            folder: The folder to record the data. Must exist.
            **kwargs: Initial pairs of NamedTuple fields, precisely as in ``NamedTuple(**kwargs)``.
                Missing fields receive their defaults, so ``Data(folder)`` is the default parameter set.
        """
        self._folder = folder if folder.exists() else self.empty(folder)
        kwargs = self.NamedTuple(**kwargs)._asdict()
        self._frames = None
        self.replace(**kwargs)

    @classmethod
    def read(cls, folder: Path | str, **kwargs: Matrix) -> Self:
        """ Read ``Data`` from ``folder``.

        Args:
            folder: The folder to record the data. Must exist
            **kwargs: key=ordinate initial pairs of NamedTuple fields, precisely as in NamedTuple(**kwargs).
                Missing fields receive their defaults, so ``Data(folder)`` is the default ``Data``.
        Returns: The ``Data`` stored in ``folder``.
        """
        folder = Path(folder)
        asdict = {field: Frame(folder / field, kwargs.get(field, None)) for field in cls.fields}
        return cls(folder, **asdict)

    @staticmethod
    def delete(folder: Path | str) -> Path:
        """ Returns a non-existent ``folder``."""
        folder = Path(folder)
        shutil.rmtree(folder, ignore_errors=True)
        return folder

    @staticmethod
    def empty(folder: Path | str) -> Path:
        """ Returns an empty ``folder``."""
        folder = Data.delete(folder)
        folder.mkdir(mode=0o777, parents=True, exist_ok=False)
        return folder

    @staticmethod
    def copy(src_folder: Path | str, dst_folder: Path | str) -> Path:
        """ Returns a copy of ``src_folder`` at dst_folder, deleting anything existing at the destination."""
        dst_folder = Data.delete(dst_folder)
        shutil.copytree(src=src_folder, dst=dst_folder)
        return dst_folder


class Model(ABC):
    """ Abstract base class for any model. This base class implements generic file storage and parameter handling.
    The latter is dealt with by each subclass overriding ``Data.NamedTuple`` with its own ``NamedTuple[NamedTuple]``
    defining the parameter set it takes. ``model.data.values`` is a ``Model.Data.NamedTuple`` of NP.Matrices.

    A Model also may include a calibrate method taking meta stored in an meta.json file, which default to cls.META.
    """

    class Data(Data):
        """ This is a placeholder which must be overridden in any implementation."""

        class NamedTuple(NamedTuple):
            """ A NamedTuple of data. Must be overridden."""
            NotImplemented: Data.Matrix = np.atleast_2d('NotImplemented')  #: NamedTuple can have any number of members.

    @classmethod
    @property
    def META(cls) -> Dict[str, Any]:
        """Returns: Default meta data."""
        pass
        # raise NotImplementedError

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def data(self) -> Data:
        return self._data

    @data.setter
    def data(self, value: Data):
        self._data = value

    @abstractmethod
    def calibrate(self, method: str, **kwargs) -> Dict[str, Any]:
        if method != 'I know I told you never to call me, but I have relented because I just cannot live without you sweet-cheeks.':
            raise NotImplementedError('base.calibrate() must never be called.')
        else:
            meta = self.META | kwargs
            meta = (meta if meta is not None
                       else self.read_meta() if self._meta_json.exists() else self.META)
            meta.pop('result', default=None)
            meta = {**meta, 'result': 'OPTIMIZE HERE !!!'}
            self.write_meta(meta)
            self.data = self._data.replace('WITH OPTIMAL PARAMETERS!!!').write(self.folder)   # Remember to write optimization results.
        return meta

    def read_meta(self) -> Dict[str, Any]:
        # noinspection PyTypeChecker
        with open(self._meta_json, mode='r') as file:
            return json.load(file)

    def write_meta(self, meta: Dict[str, Any]):
        # noinspection PyTypeChecker
        with open(self._meta_json, mode='w') as file:
            json.dump(meta, file, indent=8)

    def __repr__(self) -> str:
        """ Returns the folder path."""
        return str(self._folder)

    def __str__(self) -> str:
        """ Returns the folder name."""
        return self._folder.name

    @abstractmethod
    def __init__(self, folder: Path | str, read_data: bool = False, **kwargs: NP.Matrix):
        """ Model constructor, to be called by all subclasses as a matter of priority.

        Args:
            folder: The model file location.
            read_data: If True, the ``model.data`` are read from ``folder``, otherwise defaults are used.
            **kwargs: The model.data fields=values to replace after reading from file/defaults.
        """
        self._folder = Path(folder)
        self._meta_json = self._folder / "meta.json"
        if read_data:
            self._data = self.Data.read(self._folder).replace(**kwargs)
        else:
            self._folder.mkdir(mode=0o777, parents=True, exist_ok=True)
            self._data = self.Data(self._folder, **kwargs)
        self._implementation = None
