#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2022 Robert A. Milton. All rights reserved.
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
import copy

from romcomma.base.definitions import *
from romcomma.data.storage import Repository, Fold
import romcomma
from time import time
from datetime import timedelta
from contextlib import contextmanager
from shutil import rmtree

@contextmanager
def Timing(name: str):
    """ Context Manager for timing operations.

    Args:
        name: The name of this context, this appears as what is being timed. The empty string will not be timed.
    """
    _enter = time()
    if name != '':
        print(f'Running {name}', end='')
    yield
    if name != '':
        _exit = time()
        print(f' took {timedelta(seconds=int(_exit-_enter))}.')


@contextmanager
def Context(name: str, device: str = '', **kwargs):
    """ Context Manager for running operations.

    Args:
        name: The name of this context, this appears as what is being run.
        device: The device to run on. If this ends in the regex ``[C,G]PU*`` then the logical device ``/[C,G]*`` is used,
            otherwise device allocation is automatic.
        **kwargs: Is passed straight to the implementation GPFlow manager. Note, however, that ``float=float32`` is inoperative due to sicpy.
    """
    with Timing(name):
        kwargs = kwargs | {'float': 'float64'}
        print(' using GPFlow(' + ', '.join([f'{k}={v!r}' for k, v in kwargs.items()]), end=')')
        device = '/' + device[max(device.rfind('CPU'), device.rfind('GPU')):]
        if len(device) > 3:
            device_manager = tf.device(device)
            print(f' on {device}', end='')
        else:
            device_manager = Timing('')
        implementation_manager = gf.config.as_context(gf.config.Config(**kwargs))
        print('...')
        with device_manager:
            with implementation_manager:
                yield
        print('...Running ' + name, end='')


def gpr(name: str, repo: Repository, is_read: Optional[bool], is_isotropic: Optional[bool], is_independent: Optional[bool],
        kernel_parameters: Optional[romcomma.gpr.kernels.Kernel.Parameters] = None, parameters: Optional[romcomma.gpr.models.GP.Parameters] = None,
        optimize: bool = True, test: bool = True, **kwargs):
    """ Service routine to recursively run GPs the Folds in a Repository, or on a single Fold.

    Args:
        name: The GP name.
        repo: The source of the training data.csv. May be a Fold, or a Repository which contains Folds.
        is_read: If True, the GP.kernel.parameters and GP.parameters are read from ``fold.folder/name``, otherwise defaults are used.
            If None, the nearest available GP down the hierarchy is broadcast, constructing from scratch if no nearby GP is available.
        is_isotropic: Whether to coerce the kernel to be isotropic. If None, isotropic is run, then broadcast to run anisotropic.
        is_independent: Whether the outputs are independent of each other or not. If None, independent is run then broadcast to run dependent.
        kernel_parameters: A base.Kernel.Parameters to use for GP.kernel.parameters. If not None, this replaces the kernel specified by file/defaults.
            If None, the kernel is read from file, or set to the default base.Kernel.Parameters(), according to read_from_file.
        parameters: The GP.parameters fields=values to replace after reading from file/defaults.
        optimize: Whether to optimize each GP.
        test: Whether to test_data each GP.
        kwargs: A Dict of implementation-dependent optimizer options, similar to (and documented in) base.GP.OPTIMIZER_OPTIONS.

    Raises:
        FileNotFoundError: If repo is not a Fold, and contains no Folds.
    """
    if not isinstance(repo, Fold):
        for k in repo.folds:
            gpr(name, Fold(repo, k), is_read, is_isotropic, is_independent, kernel_parameters, parameters, optimize, test, **kwargs)
    else:
        if is_independent is None:
            gpr(name, repo, is_read, True if is_isotropic is True else None, True, kernel_parameters, parameters, optimize, test, **kwargs)
            gpr(name, repo, None, is_isotropic, False, kernel_parameters, parameters, optimize, test, **kwargs)
        else:
            full_name = name + ('.i' if is_independent else '.d')
            if is_isotropic is None:
                gpr(name, repo, is_read, True, is_independent, kernel_parameters, parameters, optimize, test, **kwargs)
                gpr(name, repo, None, False, is_independent, kernel_parameters, parameters, optimize, test, **kwargs)
            else:
                full_name = full_name + ('.i' if is_isotropic else '.a')
                if is_read is None:
                    if not (repo.folder / full_name).exists():
                        nearest_name = name + '.i' + full_name[-2:]
                        if is_independent or not (repo.folder / nearest_name).exists():
                            nearest_name = full_name[:-2] + '.i'
                            if not (repo.folder / nearest_name).exists():
                                gpr(name, repo, False, is_isotropic, is_independent, kernel_parameters, parameters, optimize, test, **kwargs)
                                return
                        romcomma.gpr.models.GP.copy(src_folder=repo.folder/nearest_name, dst_folder=repo.folder/full_name)
                    gpr(name, repo, True, is_isotropic, is_independent, kernel_parameters, parameters, optimize, test, **kwargs)
                else:
                    with Timing(f'fold.{repo.meta["k"]}, is_isotropic={is_isotropic}, is_independent={is_independent}'):
                        gp = romcomma.gpr.models.GP(full_name, repo, is_read, is_isotropic, is_independent, kernel_parameters,
                                           **({} if parameters is None else parameters.as_dict()))
                        if optimize:
                            gp.optimize(**kwargs)
                        if test:
                            gp.test()


def gpr2(name: str, repo: Repository, is_read: Optional[bool], is_isotropic: Optional[bool], is_independent: Optional[bool], broadcast_fraction: float,
        kernel_parameters: Optional[romcomma.gpr.kernels.Kernel.Parameters] = None, parameters: Optional[romcomma.gpr.models.GP.Parameters] = None,
        optimize: bool = True, test: bool = True, **kwargs):
    """ Service routine to recursively run GPs the Folds in a Repository, or on a single Fold.

    Args:
        name: The GP name.
        repo: The source of the training data.csv. May be a Fold, or a Repository which contains Folds.
        is_read: If True, the GP.kernel.parameters and GP.parameters are read from ``fold.folder/name``, otherwise defaults are used.
            If None, the nearest available GP down the hierarchy is broadcast, constructing from scratch if no nearby GP is available.
        is_isotropic: Whether to coerce the kernel to be isotropic. If None, isotropic is run, then broadcast to run anisotropic.
        is_independent: Whether the outputs are independent of each other or not. If None, independent is run then broadcast to run dependent.
        kernel_parameters: A base.Kernel.Parameters to use for GP.kernel.parameters. If not None, this replaces the kernel specified by file/defaults.
            If None, the kernel is read from file, or set to the default base.Kernel.Parameters(), according to read_from_file.
        parameters: The GP.parameters fields=values to replace after reading from file/defaults.
        optimize: Whether to optimize each GP.
        test: Whether to test_data each GP.
        kwargs: A Dict of implementation-dependent optimizer options, similar to (and documented in) base.GP.OPTIMIZER_OPTIONS.

    Raises:
        FileNotFoundError: If repo is not a Fold, and contains no Folds.
    """
    if not isinstance(repo, Fold):
        for k in repo.folds:
            gpr2(name, Fold(repo, k), is_read, is_isotropic, is_independent, broadcast_fraction, kernel_parameters, parameters, optimize, test, **kwargs)
    else:
        if is_independent is None:
            gpr2(name, repo, is_read, True if is_isotropic is True else None, True, broadcast_fraction, kernel_parameters, parameters, optimize, test, **kwargs)
            gpr2(name, repo, None, is_isotropic, False, broadcast_fraction, kernel_parameters, parameters, optimize, test, **kwargs)
        else:
            full_name = name + ('.i' if is_independent else '.d')
            if is_isotropic is None:
                gpr2(name, repo, is_read, True, is_independent, broadcast_fraction, kernel_parameters, parameters, optimize, test, **kwargs)
                gpr2(name, repo, None, False, is_independent, broadcast_fraction, kernel_parameters, parameters, optimize, test, **kwargs)
            else:
                full_name = full_name + ('.i' if is_isotropic else '.a')
                if is_read is None:
                    if not (repo.folder / full_name).exists():
                        nearest_name = name + '.i' + full_name[-2:]
                        if is_independent or not (repo.folder / nearest_name).exists():
                            nearest_name = full_name[:-2] + '.i'
                            if not (repo.folder / nearest_name).exists():
                                gpr2(name, repo, False, is_isotropic, is_independent, broadcast_fraction, kernel_parameters, parameters, optimize, test, **kwargs)
                                return
                        romcomma.gpr.models.GP.copy(src_folder=repo.folder/nearest_name, dst_folder=repo.folder/full_name)
                    gpr2(name, repo, True, is_isotropic, is_independent, broadcast_fraction, kernel_parameters, parameters, optimize, test, **kwargs)
                else:
                    gp = romcomma.gpr.models.GP(full_name, repo, is_read, is_isotropic, True, kernel_parameters,
                                       **({} if parameters is None else parameters.as_dict()))
                    gp.broadcast_parameters2(False, is_isotropic, broadcast_fraction=broadcast_fraction)
                    if optimize:
                        gp.optimize(**kwargs)
                    if test:
                        gp.test()


def gsa(name: str, repo: Repository, is_independent: bool, ignore_exceptions: bool = False, **kwargs):
    """ Service routine to recursively run GSAs on the Folds in a Repository, or on a single Fold.

    Args:
        name: The GP name.
        repo: The source of the training data.csv. May be a Fold, or a Repository which contains Folds.
        is_independent: Whether the gp kernel for each output is independent of the other outputs.
        ignore_exceptions: Whether to ignore exceptions (e.g. file not found) when they are encountered, or halt.
        kwargs: A Dict of gsa calculation options, which updates the default gsa.perform.GSA.OPTIONS.
    Raises:
        FileNotFoundError: If repo is not a Fold, and contains no Folds.
    """
    if not isinstance(repo, Fold):
        for k in repo.folds:
            gsa(name, Fold(repo, k), is_independent, ignore_exceptions, **kwargs)
    else:
        with Timing(f'fold{repo.meta["k"]}'):
            try:
                if is_independent:
                    gp = romcomma.gpr.models.GP(f'{name}.i.a' , repo, is_read=True, is_isotropic=False, is_independent=True)
                else:
                    gp = romcomma.gpr.models.GP(f'{name}.d.a', repo, is_read=True, is_isotropic=False, is_independent=False)
                romcomma.gsa.perform.GSA(gp, romcomma.gsa.perform.GSA.Kind.FIRST_ORDER, m=-1, **kwargs)
            except BaseException as exception:
                if not ignore_exceptions:
                    raise exception
                else:
                    pass


def aggregate(aggregators: Dict[str, Sequence[Dict[str, Any]]], dst: Union[Path, str], ignore_missing: bool=False, **kwargs):
    """ Aggregate csv files over aggregators.

    Args:
        aggregators: A Dict of aggregators, keyed by csv filename. An aggregator is a List of Dicts containing source folder ['folder']
            and {key: value} to insert column 'key' and populate it with 'value' in folder/csv.
        dst: The destination folder, to which csv files listed as the keys in aggregators.
        **kwargs: Read options passed directly to pd.read_csv().
    """
    dst = Path(dst)
    rmtree(dst, ignore_errors=True)
    dst.mkdir(mode=0o777, parents=True, exist_ok=False)
    for csv, aggregator in aggregators.items():
        is_initial = True
        results = None
        for file in aggregator:
            file = copy.deepcopy(file)
            filepath = Path(file.pop('folder'))/csv
            if filepath.exists() or not ignore_missing:
                result = pd.read_csv(filepath, **kwargs)
                for key, value in file.items():
                    result.insert(0, key, np.full(result.shape[0], value), True)
                if is_initial:
                    results = result.copy(deep=True)
                    is_initial = False
                else:
                    results = pd.concat([results, result.copy(deep=True)], axis=0, ignore_index=True)
        results.to_csv(dst/csv, index=False, float_format='%.6f')
