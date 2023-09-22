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

""" **User interface to GPR, GSA and ROM** """

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.data.storage import Repository, Fold
from romcomma.gpr.kernels import Kernel
from romcomma.gpr.models import GPR, MOGP
from romcomma.gsa.models import GSA, Sobol
from romcomma.user import contexts, results
import shutil


def gpr(name: str, repo: Repository, is_read: bool | None, is_covariant: bool | None, is_isotropic: bool | None, ignore_exceptions: bool = False,
        kernel_parameters: Kernel.Data | None = None, likelihood_variance: NP.Matrix | None = None,
        calibrate: bool = True, test: bool = True, **kwargs) -> List[str]:
    """ Undertake GPR on a Fold, or recursively across the Folds in a Repository.

    Args:
        name: The MOGP name.
        repo: A Fold to house the MOGP, or a Repository containing Folds to house the GPs.
        is_read: If True, MOGP kernel data and likelihood_variance are read from ``fold.folder/name``, otherwise defaults are used.
            If None, the nearest ancestor MOGP in the independence/isotropy hierarchy is recursively constructed from its nearest ancestor MOGP if necessary,
            then read and broadcast available.
        is_covariant: Whether the outputs are independent of each other or not. If None, independent is run then broadcast to run dependent.
        is_isotropic: Whether the kernel is isotropic. If None, isotropic is run, then broadcast to run anisotropic.
        ignore_exceptions: Whether to continue when the MOGP provider throws an exception.
        kernel_parameters: If not None, this replaces the Kernel specified by the MOGP default.
        likelihood_variance: If not None this replaces the likelihood_variance specified by the MOGP default.
        calibrate: Whether to calibrate each MOGP.
        test: Whether to test_data each MOGP.
        kwargs: A Dict of implementation-dependent passes straight to MOGP.Optimize().
    Returns:
        A list of the names of the GPs which have been constructed. The MOGP.Data are ``user.results.Aggregated`` over folds
    Raises:
        FileNotFoundError: If repo is not a Fold, and contains no Folds.
    """
    if not isinstance(repo, Fold):
        for k in repo.folds:
            names = gpr(name, Fold(repo, k), is_read, is_covariant, is_isotropic, ignore_exceptions, kernel_parameters, likelihood_variance, calibrate, test, **kwargs)
        if test:
            results.Collect({'test': {'header': [0, 1]}, 'test_summary': {'header': [0, 1], 'index_col': 0}},
                      {name: {} for name in names}, ignore_exceptions).from_folds(repo, True)
        results.Collect({'variance': {}, 'log_marginal': {}}, {f'{name}/likelihood': {} for name in names}, ignore_exceptions).from_folds(repo, True)
        results.Collect({'variance': {}, 'lengthscales': {}}, {f'{name}/kernel': {} for name in names}, ignore_exceptions).from_folds(repo, True)
        return names
    else:
        if is_covariant is None:
            names = gpr(name, repo, is_read, False, is_isotropic, ignore_exceptions, kernel_parameters, likelihood_variance, calibrate, test, **kwargs)
            return (names +
                    gpr(name, repo, None, True, False if is_isotropic is None else is_isotropic, ignore_exceptions,
                        kernel_parameters, likelihood_variance, calibrate, test, **kwargs))
        full_name = name + ('.c' if is_covariant else '.v')
        if is_isotropic is None:
            names = gpr(name, repo, is_read, is_covariant, True, ignore_exceptions, kernel_parameters, likelihood_variance, calibrate, test, **kwargs)
            return names + gpr(name, repo, None, is_covariant, False, ignore_exceptions, kernel_parameters, likelihood_variance, calibrate, test, **kwargs)
        full_name = full_name + ('.i' if is_isotropic else '.a')
        if is_read is None:
            if not (repo.folder / full_name).exists():
                nearest_name = name + '.v' + full_name[-2:]
                if not (is_covariant and (repo.folder / nearest_name).exists()):
                    nearest_name = full_name[:-2] + '.i'
                    if not (repo.folder / nearest_name).exists():
                        return gpr(name, repo, False, is_covariant, is_isotropic, ignore_exceptions, kernel_parameters, likelihood_variance,
                                   calibrate, test, **kwargs)
                GPR.Data.copy(src_folder=repo.folder / nearest_name, dst_folder=repo.folder / full_name)
            return gpr(name, repo, True, is_covariant, is_isotropic, ignore_exceptions, kernel_parameters, likelihood_variance, calibrate, test, **kwargs)
        with contexts.Timer(f'fold.{repo.meta["k"]} {full_name} MOGP Regression'):
            try:
                if is_read:
                    gp = MOGP(full_name, repo, is_read, is_covariant, is_isotropic)
                else:
                    gp = MOGP(full_name, repo, is_read, is_covariant, is_isotropic, kernel_parameters, likelihood_variance)
                if calibrate:
                    gp.calibrate(**kwargs)
                if test:
                    gp.test()
            except BaseException as exception:
                if not ignore_exceptions:
                    raise exception
        return [full_name]


def gsa(name: str, repo: Repository, is_covariant: Optional[bool], is_isotropic: Optional[bool],
        kinds: GSA.Kind | Sequence[GSA.Kind] = GSA.ALL_KINDS, m: int = -1,
        ignore_exceptions: bool = False, is_error_calculated: bool = False, **kwargs) -> List[Path]:
    """ Undertake GSA on a Fold, or recursively across the Folds in a Repository.

    Args:
        name: The GSA name.
        repo: A Fold to house the GSA, or a Repository containing Folds to house the GSAs.
        is_covariant: Whether each output is independent of the other outputs. None results in variant (independent) followed by covariant (dependent).
        is_isotropic: Whether the kernel is isotropic. If None, isotropic is run, then broadcast to run anisotropic.
        kinds: Kind of index to calculate - first_order, closed or total. A Sequence of Kinds will be run consecutively.
        is_error_calculated: Whether to calculate variances (errors) on the Sobol indices.
            The calculation of error is memory intensive, so leave this flag as False unless you are sure you need errors.
            Furthermore, errors will only be calculated if the kernel of the GP has diagonal variance F.
        m: The dimensionality of the reduced model. For a single calculation it is required that ``0 < m < gp.M``.
            Any m outside this range results the Sobol index of each kind being calculated for all ``m in range(1, M+1)``.
        ignore_exceptions: Whether to ignore exceptions (e.g. file not found) when they are encountered, or halt.
        kwargs: A Dict of gsa calculation options, which updates the default gsa.undertake.calculation.META.
    Raises:
        FileNotFoundError: If repo is not a Fold, and contains no Folds.
    Returns:
        A list of the calculation names which have been run, relative to repo.folder.
    """
    kinds = (kinds,) if isinstance(kinds, GSA.Kind) else kinds
    if not isinstance(repo, Fold):
        names = []
        for k in repo.folds:
            names = gsa(name, Fold(repo, k), is_covariant, is_isotropic, kinds, m, ignore_exceptions, is_error_calculated, **kwargs)
        results.Collect({'S': {}, 'V': {}} | ({'T': {}, 'W': {}} if is_error_calculated else {}),
                        {name: {} for name in names}, ignore_exceptions).from_folds(repo, True)
        for name in names:
            shutil.copyfile(repo.fold_folder(repo.folds.start) / 'meta.json', repo.folder / name / 'meta.json')
    else:
        if is_covariant is None:
            names = gsa(name, repo, False, is_isotropic, kinds, m, ignore_exceptions, is_error_calculated, **kwargs)
            return (names +
                    gsa(name, repo, True, False if is_isotropic is None else is_isotropic, kinds, m, ignore_exceptions, is_error_calculated, **kwargs))
        full_name = name + ('.c' if is_covariant else '.v')
        if is_isotropic is None:
            names = gsa(name, repo, is_covariant, True, kinds, m, ignore_exceptions, is_error_calculated, **kwargs)
            return names + gsa(name, repo, is_covariant, False, kinds, m, ignore_exceptions, is_error_calculated, **kwargs)
        full_name = full_name + ('.i' if is_isotropic else '.a')
        with contexts.Timer(f'fold.{repo.meta["k"]} {full_name} GSA'):
            names = []
            try:
                gp = MOGP(full_name, repo, is_read=True, is_covariant=is_covariant, is_isotropic=is_isotropic)
                for kind in kinds:
                    folder = Sobol(gp, kind, m, is_error_calculated, **kwargs).calibrate().get('folder')
                    names += [Path(folder).relative_to(repo.folder)]
            except BaseException as exception:
                if not ignore_exceptions:
                    raise exception
    return names
