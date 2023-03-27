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

""" Contains base classes for romcomma Models and Parameters."""

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.data.storage import Frame
import shutil
import json
from abc import ABC


# noinspection PyProtectedMember
class Parameters(ABC):
    """ Abstraction of the parameters of a Model. Essentially a NamedTuple backed by files in a folder.
    Note that file writing is lazy, it must be called explicitly, but the Parameters are designed for chained calls.

    Most Parameters methods are simple wrappers for annoyingly underscored methods of `NamedTuple
    <https://docs.python.org/3/library/collections.html#collections.namedtuple>`_."""

    @classmethod
    @property
    @abstractmethod
    def Values(cls) -> Type[NamedTuple]:
        """ The NamedTuple underpinning this Parameters set."""

    @classmethod
    def make(cls, iterable: Iterable) -> Parameters:
        return cls.Values._make(iterable)

    @classmethod
    @property
    def fields(cls) -> Tuple[str, ...]:
        return cls.Values._fields

    @classmethod
    @property
    def field_defaults(cls) -> Dict[str, Any]:
        return cls.Values._field_defaults

    def as_dict(self) -> Dict[str, Any]:
        return self._values._asdict()

    def replace(self, **kwargs: NP.Matrix) -> Parameters:
        for key, value in kwargs.items():
            kwargs[key] = np.atleast_2d(value)
        self._values = self._values._replace(**kwargs)
        return self

    @property
    def folder(self) -> Path:
        return self._folder

    @folder.setter
    def folder(self, value: Optional[PathLike]):
        if value is not None:
            self._folder = Path(value)
            self._csvs = tuple((self._folder / field).with_suffix(".csv") for field in self.fields)

    def csv(self, field: str) -> Path:
        assert getattr(self, '_csvs', None) is not None, 'Cannot perform file operations before self._folder and self._csvs are set.'
        i = self.fields.index(field)
        return self._csvs[i]

    @property
    def values(self) -> Values:
        return self._values

    @values.setter
    def values(self, value: Values):
        self._values = self.Values(*(np.atleast_2d(val) for val in value))

    def broadcast_value(self, model_name: str, field: str, target_shape: Tuple[int, int], is_diagonal: bool = True,
                        folder: Optional[PathLike] = None) -> Parameters:
        """ Broadcast a parameter ordinate.

        Args:
            model_name: Used only in error reporting.
            field: The name of the field whose ordinate we are broadcasting.
            target_shape: The shape to broadcast to.
            is_diagonal: Whether to zero the off-diagonal elements of a square matrix.
            folder:

        Returns: Self, for chaining calls.
        Raises:
            IndexError: If broadcasting is impossible.
        """
        replacement = {field: getattr(self.values, field)}
        try:
            replacement[field] = np.array(np.broadcast_to(replacement[field], target_shape))
        except ValueError:
            raise IndexError(f'The {model_name} {field} has shape {replacement[field].shape} '
                             f' which cannot be broadcast to {target_shape}.')
        if is_diagonal and target_shape[0] > 1:
            replacement[field] = np.diag(np.diagonal(replacement[field]))
        return self.replace(**replacement).write(folder)

    def read(self) -> Parameters:
        """ Read Parameters from their csv files.

        Returns: ``self``, for chaining calls.
        Raises:
            AssertionError: If self.csv is not set.
        """
        assert getattr(self, '_csvs', None) is not None, 'Cannot perform file operations before self._folder and self._csvs are set.'
        self._values = self.Values(**{key: Frame(self._csvs[i], header=[0]).df.values for i, key in enumerate(self.fields)})
        return self

    def write(self, folder: Optional[PathLike] = None) -> Parameters:
        """  Write Parameters to their csv files.

        Args:
            folder: The file location is changed to ``folder`` unless ``folder`` is ``None`` (the default).
        Returns: ``self``, for chaining calls.
        Raises:
            AssertionError: If self.csv is not set.
        """
        self.folder = folder
        assert getattr(self, '_csvs', None) is not None, 'Cannot perform file operations before self._folder and self._csvs are set.'
        tuple(Frame(self._csvs[i], pd.DataFrame(p)) for i, p in enumerate(self._values))
        return self

    def __repr__(self) -> str:
        return str(self._folder)

    def __str__(self) -> str:
        return self._folder.name

    def __init__(self, folder: Optional[PathLike] = None, **kwargs: NP.Matrix):
        """ Parameters Constructor. Shouldn't need to be overridden. Does not write to file.

        Args:
            folder: The folder to record the parameters.
            **kwargs: key=ordinate initial pairs of NamedTuple fields, precisely as in NamedTuple(**kwargs). It is the caller's responsibility to ensure
                that every ordinate is of type NP.Matrix. Missing fields receive their defaults, so Parameters(folder) is the default parameter set.
        """
        for key, value in kwargs.items():
            kwargs[key] = np.atleast_2d(value)
        self.folder = folder
        self._values = self.Values(**kwargs)


class Model(ABC):
    """ Abstract base class for any model. This base class implements generic file storage and parameter handling.
    The latter is dealt with by each subclass overriding ``Parameters.Values`` with its own ``Type[NamedTuple]``
    defining the parameter set it takes. ``model.parameters.values`` is a ``Model.Parameters.Values`` of NP.Matrices.

    A Model also may include an optimize method taking options stored in an options.json file, which default to cls.OPTIONS.
    """

    class Parameters(Parameters):
        """ This is a placeholder which must be overridden in any implementation."""

        @classmethod
        @property
        def Values(cls) -> Type[NamedTuple]:

            class Values(NamedTuple):
                """ The parameters set of a Model."""
                NotImplemented: NP.Matrix = np.atleast_2d('Not Implemented')

            return Values

    @staticmethod
    def delete(folder: PathLike, ignore_errors: bool = True):
        shutil.rmtree(folder, ignore_errors=ignore_errors)

    @staticmethod
    def copy(src_folder: PathLike, dst_folder: PathLike, ignore_errors: bool = True):
        shutil.rmtree(dst_folder, ignore_errors=ignore_errors)
        shutil.copytree(src=src_folder, dst=dst_folder)

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        """Returns: Default options."""
        pass
        # raise NotImplementedError

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    @parameters.setter
    def parameters(self, value: Parameters):
        self._parameters = value

    @property
    def params(self) -> Parameters.Values:
        """ A shorthand to save typing.

        Returns: ``self._parameters.values``
        """
        return self._parameters.values

    def optimize(self, method: str, **kwargs) -> Dict[str, Any]:
        if method != 'I know I told you never to call me, but I have relented because I just cannot live without you sweet-cheeks.':
            raise NotImplementedError('base.optimize() must never be called.')
        else:
            options = self.OPTIONS | kwargs
            options = (options if options is not None
                       else self._read_options() if self._options_json.exists() else self.OPTIONS)
            options.pop('result', default=None)
            options = {**options, 'result': 'OPTIMIZE HERE !!!'}
            self._write_options(options)
            self.parameters = self._parameters.replace('WITH OPTIMAL PARAMETERS!!!').write(self.folder)   # Remember to write optimization results.
        return options

    def _read_options(self) -> Dict[str, Any]:
        # noinspection PyTypeChecker
        with open(self._options_json, mode='r') as file:
            return json.load(file)

    def _write_options(self, options: Dict[str, Any]):
        # noinspection PyTypeChecker
        with open(self._options_json, mode='w') as file:
            json.dump(options, file, indent=8)

    def __repr__(self) -> str:
        """ Returns the folder path."""
        return str(self._folder)

    def __str__(self) -> str:
        """ Returns the folder name."""
        return self._folder.name

    @abstractmethod
    def __init__(self, folder: PathLike, read_parameters: bool = False, **kwargs: NP.Matrix):
        """ Model constructor, to be called by all subclasses as a matter of priority.

        Args:
            folder: The model file location.
            read_parameters: If True, the ``model.parameters`` are read from ``folder``, otherwise defaults are used.
            **kwargs: The model.parameters fields=values to replace after reading from file/defaults.
        """
        self._folder = Path(folder)
        self._options_json = self._folder / "options.json"

        if read_parameters:
            self._parameters = self.Parameters(self._folder).read().replace(**kwargs)
        else:
            self._folder.mkdir(mode=0o777, parents=True, exist_ok=True)
            self._parameters = self.Parameters(self._folder).replace(**kwargs)
        self._parameters.write()
        self._implementation = None
