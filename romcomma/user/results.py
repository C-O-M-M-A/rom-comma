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

""" **Functionality for processing results generated by ``romcomma``** """

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.base.classes import Data
from romcomma.data.storage import Repository, Fold
from shutil import rmtree


def copy(src: Path | str, dst: Path | str) -> Path:
    """ Copy a folder destructively.

    Args:
        src: The folder to be copied, relative to the ``Fold.folder``.
        dst: The folder of the copy, relative to the ``Fold.folder``.

    Returns: dst if successful.
    """
    Data.copy(src, dst)
    return dst


class Collect:
    """ A device for collecting -- i.e. concatenating -- csv files across folders or folds."""

    csvs: Dict[str, Dict[str, Any]] = {}    #: Key = csv name (minus extension). Value = a Dict of options (kwargs) passed to ``pd.read_csv``.
    folders: Dict[str, Dict[str, Any]] = {} #: Key = folder containing csvs. Value = An (ordered) Dict of {Column name: Column value} to insert from R to L.
    ignore_missing: bool = False    #: Whether to raise an exception when a csv is missing from a folder.
    write_options: Dict[str, Any] = {'index': False, 'float_format': '%.6f'}    #: kwargs passed straight to ``pd.to_csv``.

    def __call__(self, dst: Union[Repository, Path, str], is_existing_deleted=False, **kwargs: Any):
        """ Collect ``self.csvs`` into ``dst``. If and only if ``dst`` is a Repository, ``self.over_folds`` is called instead of ``self.over_folders``.

        Args:
            dst: The destination folder, to house ``self.csvs`` or ``self.folders``.
            is_existing_deleted: Whether to delete and recreate an existing ``dst``.
            **kwargs:  Write options passed straight to ``pd.to_csv``.
        """
        if isinstance(dst, Repository):
            return self.from_folds(dst, is_existing_deleted, **kwargs)
        else:
            return self.from_folders(dst, is_existing_deleted, **kwargs)

    def from_folders(self, dst: Union[Path, str], is_existing_deleted=False, **kwargs: Any) -> Collect:
        """ Collect ``dst/[self.csvs]`` from ``self.folders``.

        Args:
            dst: The destination folder, to house ``[self.csvs]``.
            is_existing_deleted: Whether to delete and recreate an existing ``dst``.
            **kwargs:  Write options passed straight to ``pd.to_csv``.

        Returns: ``self'' for chaining calls.
        """
        dst = Path(dst)
        if is_existing_deleted:
            rmtree(dst, ignore_errors=True)
        dst.mkdir(mode=0o777, parents=True, exist_ok=True)
        for csv, read_options in self.csvs.items():
            is_initial = True
            results = None
            for folder, columns in self.folders.items():
                file = Path(folder) / f'{csv}.csv'
                if file.exists() or not self.ignore_missing:
                    result = pd.read_csv(file, **read_options)
                    for key, value in columns.items():
                        result.insert(0, key, np.full(result.shape[0], value), True)
                    if is_initial:
                        results = result.copy(deep=True)
                        is_initial = False
                    else:
                        results = pd.concat([results, result.copy(deep=True)], axis=0, ignore_index=True)
            if not (results is None and self.ignore_missing):
                results.to_csv(dst / f'{csv}.csv', **(self.write_options | kwargs))
        return self

    def from_folds(self, dst: Repository, is_existing_deleted=False, **kwargs: Any) -> Collect:
        """ Collect ``dst/[self.folders]`` from ``Fold(dst, [k])/[self.folders]`` for ``k in self.Folds``.

        Args:
            dst: The destination folder, to house ``[self.folders]``.
            is_existing_deleted: Whether to delete and recreate an existing ``dst``.
            **kwargs:  Write options passed straight to ``pd.to_csv``.

        Returns: ``self'' for chaining calls.
        """
        if isinstance(dst, Fold):
            raise NotADirectoryError('dst is a Fold, which cannot contain other Folds, so cannot be Collected from.')
        folds = tuple((Fold(dst, k) for k in dst.folds))
        for sub_folder, extra_columns in self.folders.items():
            folders = {fold.folder / sub_folder: {'fold': fold.meta['k'], 'N': fold.N} | extra_columns for fold in folds}
            Collect(self.csvs, folders, self.ignore_missing).from_folders(dst.folder / sub_folder, is_existing_deleted, **kwargs)
        return self

    def __init__(self, csvs: Dict[str, Dict[str, Any]] = None, folders: Dict[str, Dict[str, Any]] = None, ignore_missing: bool = False, **kwargs: Any):
        """ Construct a Collect object.

        Args:
            csvs: Key = csv name (minus extension). Value = a Dict of options (kwargs) passed to ``pd.read_csv``.
            folders: Key = folder containing csvs. Value = An (ordered) Dict of {Column name: Column value} to insert from R to L.
            ignore_missing: Whether to raise an exception when a csv is missing from a folder.
            **kwargs: kwargs passed straight to ``pd.to_csv``.
        """
        self.csvs = self.csvs if csvs is None else csvs
        self.folders = self.folders if folders is None else folders
        self.ignore_missing = ignore_missing
        self.write_options.update(kwargs)
