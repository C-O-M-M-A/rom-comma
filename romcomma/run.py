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

""" Contains routines for running models. """

from __future__ import annotations

import shutil
from romcomma.base.definitions import *
from romcomma.base.classes import Model
from romcomma.data.storage import Repository, Fold
from romcomma.gpr import models, kernels
from romcomma.gsa import perform
from time import time
from datetime import timedelta
from contextlib import contextmanager
from shutil import rmtree
from copy import deepcopy

@contextmanager
def TimingOneLiner(name: str):
    """ Context Manager for timing operations.

    Args:
        name: The name of this context, this appears as what is being timed. The empty string will not be timed.
    """
    _enter = time()
    if name != '':
        print(f'Running {name}', end='', flush=True)
    yield
    if name != '':
        _exit = time()
        print(f' took {timedelta(seconds=int(_exit-_enter))}.')


@contextmanager
def Timing(name: str):
    """ Context Manager for timing operations.

    Args:
        name: The name of this context, this appears as what is being timed. The empty string will not be timed.
    """
    _enter = time()
    if name != '':
        print(f'Running {name}...')
    yield
    if name != '':
        _exit = time()
        print(f'...took {timedelta(seconds=int(_exit-_enter))}.')


@contextmanager
def Context(name: str, device: str = '', **kwargs):
    """ Context Manager for running operations.

    Args:
        name: The name of this context, this appears as what is being run.
        device: The device to run on. If this ends in the regex ``[C,G]PU*`` then the logical device ``/[C,G]*`` is used,
            otherwise device allocation is automatic.
        **kwargs: Is passed straight to the implementation GPFlow manager. Note, however, that ``float=float32`` is inoperative due to sicpy.
    """
    with TimingOneLiner(name):
        kwargs = kwargs | {'float': 'float64'}
        eager = kwargs.pop('eager', None)
        tf.config.run_functions_eagerly(eager)
        print(' using GPFlow(' + ', '.join([f'{k}={v!r}' for k, v in kwargs.items()]), end=')')
        device = '/' + device[max(device.rfind('CPU'), device.rfind('GPU')):]
        if len(device) > 3:
            device_manager = tf.device(device)
            print(f' on {device}', end='')
        else:
            device_manager = TimingOneLiner('')
        implementation_manager = gf.config.as_context(gf.config.Config(**kwargs))
        print('...')
        with device_manager:
            with implementation_manager:
                yield
        print('...Running ' + name, end='')


def copy(src: str, dst: str, repo: Repository):
    """ Service routine to copy a model across the Folds in a Repository, or in a single Fold.

    Args:
        src: The model to be copied.
        dst: The name of the copy.
        repo: The source of the training data.csv. May be a Fold, or a Repository which contains Folds.

    Raises:
        FileNotFoundError: If repo is not a Fold, and contains no Folds.
    """
    if not isinstance(repo, Fold):
        for k in repo.folds:
            copy(src, dst, Fold(repo, k))
    else:
        Model.copy(repo.folder / src, repo.folder / dst)


def gpr(name: str, repo: Repository, is_read: Optional[bool], is_independent: Optional[bool], is_isotropic: Optional[bool], ignore_exceptions: bool = False,
        kernel_parameters: Optional[kernels.Kernel.Parameters] = None, parameters: Optional[models.GP.Parameters] = None,
        optimize: bool = True, test: bool = True, **kwargs) -> List[str]:
    """ Service routine to recursively run GPs the Folds in a Repository, or on a single Fold.

    Args:
        name: The GP name.
        repo: The source of the training data.csv. May be a Fold, or a Repository which contains Folds.
        is_read: If True, the GP.kernel.parameters and GP.parameters are read from ``fold.folder/name``, otherwise defaults are used.
            If None, the nearest available GP down the hierarchy is broadcast, constructing from scratch if no nearby GP is available.
        is_independent: Whether the outputs are independent of each other or not. If None, independent is run then broadcast to run dependent.
        is_isotropic: Whether the kernel is isotropic. If None, isotropic is run, then broadcast to run anisotropic.
        ignore_exceptions: Whether to continue when the GP provider throws an exception.
        kernel_parameters: A base.Kernel.Parameters to use for GP.kernel.parameters. If not None, this replaces the kernel specified by file/defaults.
            If None, the kernel is read from file, or set to the default base.Kernel.Parameters(), according to read_from_file.
        parameters: The GP.parameters fields=values to replace after reading from file/defaults.
        optimize: Whether to optimize each GP.
        test: Whether to test_data each GP.
        kwargs: A Dict of implementation-dependent optimizer options, similar to (and documented in) models.GP.Optimize().
    Returns:
        A list of the GP names which have been run.
    Raises:
        FileNotFoundError: If repo is not a Fold, and contains no Folds.
    """
    if not isinstance(repo, Fold):
        for k in repo.folds:
            names = gpr(name, Fold(repo, k), is_read, is_independent, is_isotropic, ignore_exceptions, kernel_parameters, parameters, optimize, test, **kwargs)
        if test:
            Aggregate({'test': {'header': [0, 1]}, 'test_summary': {'header': [0, 1], 'index_col': 0}},
                      {name: {} for name in names}, ignore_exceptions).over_folds(repo, True)
        Aggregate({'variance': {}, 'log_marginal': {}}, {f'{name}/likelihood': {} for name in names}, ignore_exceptions).over_folds(repo, True)
        Aggregate({'variance': {}, 'lengthscales': {}}, {f'{name}/kernel': {} for name in names}, ignore_exceptions).over_folds(repo, True)
        return names
    else:
        if is_independent is None:
            names = gpr(name, repo, is_read, True, is_isotropic, ignore_exceptions, kernel_parameters, parameters, optimize, test, **kwargs)
            return (names +
                    gpr(name, repo, None, False, False if is_isotropic is None else is_isotropic, ignore_exceptions,
                        kernel_parameters, parameters, optimize, test, **kwargs))
        full_name = name + ('.i' if is_independent else '.d')
        if is_isotropic is None:
            names = gpr(name, repo, is_read, is_independent, True, ignore_exceptions, kernel_parameters, parameters, optimize, test, **kwargs)
            return names + gpr(name, repo, None, is_independent, False, ignore_exceptions, kernel_parameters, parameters, optimize, test, **kwargs)
        full_name = full_name + ('.i' if is_isotropic else '.a')
        if is_read is None:
            if not (repo.folder / full_name).exists():
                nearest_name = name + '.i' + full_name[-2:]
                if is_independent or not (repo.folder / nearest_name).exists():
                    nearest_name = full_name[:-2] + '.i'
                    if not (repo.folder / nearest_name).exists():
                        return gpr(name, repo, False, is_independent, is_isotropic, ignore_exceptions, kernel_parameters, parameters, optimize, test, **kwargs)
                models.GP.copy(src_folder=repo.folder/nearest_name, dst_folder=repo.folder/full_name)
            return gpr(name, repo, True, is_independent, is_isotropic, ignore_exceptions, kernel_parameters, parameters, optimize, test, **kwargs)
        with TimingOneLiner(f'fold.{repo.meta["k"]} {full_name} GP Regression'):
            try:
                gp = models.GP(full_name, repo, is_read, is_independent, is_isotropic, kernel_parameters,
                                            **({} if parameters is None else parameters.as_dict()))
                if optimize:
                    gp.optimize(**kwargs)
                if test:
                    gp.test()
            except BaseException as exception:
                if not ignore_exceptions:
                    raise exception
        return [full_name]


def gsa(name: str, repo: Repository, is_independent: Optional[bool], is_isotropic: Optional[bool], kinds: Sequence[perform.GSA.Kind] = perform.GSA.ALL_KINDS,
        m: int = -1, ignore_exceptions: bool = False, is_error_calculated: bool = False, **kwargs) -> List[Path]:
    """ Service routine to recursively run GSAs on the Folds in a Repository, or on a single Fold.

    Args:
        name: The GP name.
        repo: The source of the training data.csv. May be a Fold, or a Repository which contains Folds.
        is_independent: Whether the gp kernel for each output is independent of the other outputs. None results in independent followed by dependent.
        is_isotropic: Whether the kernel is isotropic. If None, isotropic is run, then broadcast to run anisotropic.
        kinds: The gsa.perform.Kind of index to calculate - first order, closed or total. A Sequence of Kinds will be run consecutively.
        is_error_calculated: Whether to calculate variances (errors) on the Sobol indices.
            The calculation of is memory intensive, so leave this flag as False unless you are sure you need errors.
            Furthermore, errors will only be calculated if the kernel of the gp has diagonal variance F.
        m: The dimensionality of the reduced model. For a single calculation it is required that ``0 < m < gp.M``.
            Any m outside this range results the Sobol index of kind being calculated for all ``m in range(1, M+1)``.
        ignore_exceptions: Whether to ignore exceptions (e.g. file not found) when they are encountered, or halt.
        kwargs: A Dict of gsa calculation options, which updates the default gsa.perform.GSA.OPTIONS.
    Raises:
        FileNotFoundError: If repo is not a Fold, and contains no Folds.
    Returns:
        A list of the GSA names which have been run, relative to repo.folder.
    """
    kinds = (kinds,) if isinstance(kinds, perform.GSA.Kind) else kinds
    if not isinstance(repo, Fold):
        for k in repo.folds:
            names = gsa(name, Fold(repo, k), is_independent, is_isotropic, kinds, m, ignore_exceptions, is_error_calculated, **kwargs)
        Aggregate({'S': {}, 'V': {}} | ({'T': {}, 'W': {}} if is_error_calculated else {}),
                  {name: {} for name in names}, ignore_exceptions).over_folds(repo, True)
        for name in names:
            shutil.copyfile(repo.fold_folder(repo.folds.start) / 'meta.json', repo.folder / name / 'meta.json')
    else:
            if is_independent is None:
                names = gsa(name, repo, True, is_isotropic, kinds, m, ignore_exceptions, is_error_calculated, **kwargs)
                return (names +
                        gsa(name, repo, False, False if is_isotropic is None else is_isotropic, kinds, m, ignore_exceptions, is_error_calculated, **kwargs))
            full_name = name + ('.i' if is_independent else '.d')
            if is_isotropic is None:
                names = gsa(name, repo, is_independent, True, kinds, m, ignore_exceptions, is_error_calculated, **kwargs)
                return names + gsa(name, repo, is_independent, False, kinds, m, ignore_exceptions, is_error_calculated, **kwargs)
            full_name = full_name + ('.i' if is_isotropic else '.a')
            with TimingOneLiner(f'fold.{repo.meta["k"]} {full_name} GSA'):
                try:
                    gp = models.GP(full_name, repo, is_read=True, is_independent=is_independent, is_isotropic=is_isotropic)
                    names = []
                    for kind in kinds:
                        folder = perform.GSA(gp, kind, m, is_error_calculated, **kwargs).folder
                        names += [folder.relative_to(repo.folder)]
                except BaseException as exception:
                    if not ignore_exceptions:
                        raise exception
    return names


class Aggregate:
    """ A device for aggregating csvs across folders."""

    csvs: Dict[str, Dict[str, Any]] = {}    # Key: csv name (minus extension). Value: a Dict of options (kwargs) passed to pd.read_csv.
    folders: Dict[str, Dict[str, Any]] = {}   # Key: folder containing csvs. Value: An (ordered) Dict of {Column name: Column value} to insert in reverse order.
    ignore_missing: bool = False    # Whether to raise an exception when a csv is missing from a folder.
    write_options: Dict[str, Any] = {'index': False, 'float_format': '%.6f'}    # kwargs passed straight to pd.to_csv.

    def __call__(self, dst: Union[Repository, Path, str], delete_existing=False, **kwargs: Any):
        """ Aggregate self.csvs into dst. If dst is a repo, self.over_folds is called, otherwise self.over_folders.

        Args:
            dst: The destination folder, to house self.csvs, aggregated over self.folders.
            delete_existing: Whether to delete and recreate an existing dst.
            **kwargs:  Write options passed straight to pd.to_csv.
        """
        if isinstance(dst, Repository):
            return self.over_folds(dst, delete_existing, **kwargs)
        else:
            return self.over_folders(dst, delete_existing, **kwargs)

    def over_folders(self, dst: Union[Path, str], delete_existing=False, **kwargs: Any):
        """ Aggregate self.csvs over self.folders.

        Args:
            dst: The destination folder, to house self.csvs, aggregated over self.folders.
            delete_existing: Whether to delete and recreate an existing dst.
            **kwargs:  Write options passed straight to pd.to_csv.
        """
        dst = Path(dst)
        if delete_existing:
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
            results.to_csv(dst / f'{csv}.csv', **(self.write_options | kwargs))

    def over_folds(self, dst: Repository, delete_existing=False, **kwargs: Any):
        """ Aggregate self.csvs in dst.fold[k]/self.folders over k -- i.e. the folds of dst.

        Args:
            dst: The destination repo, to house csv files listed as the keys in aggregators.
            delete_existing: Whether to delete and recreate an existing dst.
            **kwargs:  Write options passed straight to pd.to_csv.
        """
        if isinstance(dst, Fold):
            raise NotADirectoryError('dst is a Fold, which cannot contain other Folds, so cannot be aggregated over.')
        folds = tuple((Fold(dst, k) for k in dst.folds))
        for sub_folder, extra_columns in self.folders.items():
            folders = {fold.folder / sub_folder: {'fold': fold.meta['k'], 'N': fold.N} | extra_columns for fold in folds}
            Aggregate(self.csvs, folders, self.ignore_missing).over_folders(dst.folder / sub_folder, delete_existing, **kwargs)

    def __init__(self, csvs: Dict[str, Dict[str, Any]] = None, folders: Dict[str, Dict[str, Any]] = None, ignore_missing: bool = False, **kwargs: Any):
        """ Construct an Aggregate object.

        Args:
            csvs: Key = csv name (minus extension). Value = a Dict of options (kwargs) passed to pd.read_csv.
            folders: Key = folder containing csvs. Value = An OrderedDict of {Column name: Column value} to insert (in reverse order) at left of csv.
            ignore_missing: bool: Whether to raise an exception when a csv is missing from a folder.
            **kwargs:  Write options passed straight to pd.to_csv.
        """
        self.csvs = self.csvs if csvs is None else csvs
        self.folders = self.folders if folders is None else folders
        self.ignore_missing = ignore_missing
        self.write_options.update(kwargs)
