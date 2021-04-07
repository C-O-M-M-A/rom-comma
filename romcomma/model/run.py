# BSD 3-Clause License
#
# Copyright (c) 2019-2021, Robert A. Milton
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Contains routines for running models.

**Contents**:
    **Module** enum with each value bound to an model implementation module.

    **gp** function to instantiate a GaussianBundle.

    **_gbs** function that runs Gaussian Bundles across the Folds of a store.

    **gbs** function to instantiate, optimize and test GaussianBundles on a store.

    **ROM_fold** function to instantiate a Reduced Order Model on a Fold.

    **ROM_store** function to instantiate a Reduced Order Model across the Folds of a store.

    **collect** function to instantiate the collection of new parameters.

    **collect_tests** function to instantiate the collection of test results.
"""

from ..typing_ import NP, Optional, Sequence, Union, List, Dict
from ..data import Store, Fold, Frame
from . import base
from numpy import atleast_1d, atleast_2d, full, broadcast_to, transpose
from pandas import concat
from enum import Enum
from pathlib import Path
from . import gpy_, scipy_
import time


class Module(Enum):
    """ Enumerate implementations of model.base. Each implementation is a module."""
    GPY_ = gpy_
    SCIPY_ = scipy_


def GPs(module: Module, name: str, store: Store, M: Union[int, List[int]], parameters: Optional[base.GP.Parameters],
        optimize: bool, test: bool, sobol: bool, optimizer_options: Dict = None, semi_norm: Dict = base.Sobol.SemiNorm.DEFAULT_META,
        make_ard: bool=False):
    """ Service routine to recursively run GPs on the Splits in a Store, the Folds in a Split or Store, and on a single Fold.

    Args:
        module: Sets the implementation to either Module.GPY_ or Module.SCIPY_.
        name: The GP name.
        store: The source of the training __data__.csv. May be a Fold, or a Split (whose Folds are to be analyzed),
            or a Store which contains Splits or Folds.
        M: The number of input dimensions to use. If a list is given, its length much match the number of Splits in store.
            Fold.M is actually used for the current Fold, but Folds are initialized with M
            (with the usual proviso that M is between 1 and the number of input columns in __data__.csv).
        parameters: The parameters with which to initialize each GP. If None, these are read from Fold/name in each Fold.
            If provided, follow the instructions in base.py, in particular populate parameters.kernel with the Kernel.Parameters NamedTuple
            (not numpy array!) needed to construct the kernel.
        optimize: Whether to optimize each GP.
        test: Whether to test each GP.
        sobol: Whether to calculate Sobol' indices for each GP.
        optimizer_options: A Dict of implementation-dependent optimizer options, similar to (and documented in) base.GP.DEFAULT_OPTIMIZER_OPTIONS.
        semi_norm: Meta json describing a Sobol.SemiNorm.
        make_ard: True makes a new ARD GP.
    Raises:
        IndexError: If M is a list and len(M) != len(store.splits).
        FileNotFoundError: If store is not a Fold, and contains neither Splits nor Folds.
    """
    optimizer_options = module.value.GP.DEFAULT_OPTIMIZER_OPTIONS if optimizer_options is None else optimizer_options
    splits = store.splits
    if splits:
        if isinstance(M, list):
            if len(M) != len(splits):
                raise IndexError("M has {0:d} elements which does not match the number of splits ({1:d}).".format(len(M),
                                                                                                                       len(splits)))
        else:
            M = [M] * len(splits)
        for split_index, split_dir in splits:
            split = Store(split_dir)
            GPs(module, name, split, M[split_index], parameters, optimize, test, sobol, optimizer_options, semi_norm, make_ard)
    elif not isinstance(store, Fold):
        start_time = time.time()
        K_range = range(store.meta['K'])
        if K_range:
            for k in K_range:
                GPs(module, name, Fold(store, k, M), M, parameters, optimize, test, sobol, optimizer_options, semi_norm, make_ard)
                print("Fold", k, "has finished")
        else:
            raise FileNotFoundError('Cannot construct a GP in a Store ({0:s}) which is not a Fold.'.format(store.dir))
        elapsed_mins = (time.time() - start_time) / 60
        print(store.dir.name, "has finished in {:.2f} minutes.".format(elapsed_mins))
    else:
        if make_ard:
            base.Model.copy(store.dir / name, store.dir / (name + '.ard'))
            name += '.ard'
            kernel = module.value.Kernel.ExponentialQuadratic(None, None, (store.dir / name) / module.value.GP.KERNEL_NAME)
            kernel.make_ard(store.M)
        gp = module.value.GP(fold=store, name=name, parameters=parameters)
        if optimize:
            gp.optimize(**optimizer_options)
        if test:
            gp.test()
        if sobol:
            module.value.Sobol(gp, semi_norm)


def ROMs(module: Module, name: str, store: Store, source_gp_name: str, Mu: Union[int, List[int]], Mx: Union[int, List[int]] = -1,
         optimizer_options: Dict = None, rbf_parameters: Optional[base.GP.Parameters] = None):
    """ Service routine to recursively run ROMs on the Splits in a Store, the Folds in a Split or Store, and on a single Fold.

    Args:
        module: Sets the implementation to either Module.GPY_ or Module.SCIPY_.
        name: The ROM name.
        store: The source of the training __data__.csv. May be a Fold, or a Split (whose Folds are to be analyzed),
            or a Store which contains Splits or Folds.
        source_gp_name: The name of the source GP for the ROM. Must exist in every Fold.
        Mu: The dimensionality of the rotated basis. If a list is given, its length much match the number of Splits in store.
            If Mu is not between 1 and Mx, Mx is used
            (where Mx is replaced by  the number of input columns in __data__.csv whenever Mx is not between 1 and the number of input columns in
            __data__.csv).
        Mx: The number of input dimensions to use. If a list is given, its length much match the number of Splits in store.
            Fold.M is actually used for the current Fold, but Folds are initialized with Mx
            (with the usual proviso that Mx is between 1 and the number of input columns in __data__.csv).
        optimizer_options: A Dict of implementation-dependent optimizer options, similar to (and documented in) base.ROM.DEFAULT_OPTIMIZER_OPTIONS.

    Raises:
        IndexError: If Mu is a list and len(Mu) != len(store.splits).
        IndexError: If Mx is a list and len(Mu) != len(store.splits).
        FileNotFoundError: If store is not a Fold, and contains neither Splits nor Folds.
    """
    optimizer_options = module.value.ROM.DEFAULT_OPTIMIZER_OPTIONS if optimizer_options is None else optimizer_options
    splits = store.splits
    if splits:
        if isinstance(Mx, list):
            if len(Mx) != len(splits):
                raise IndexError("Mx has {0:d} elements which does not match the number of splits ({1:d}).".format(len(Mx), len(splits)))
        else:
            Mx = [Mx] * len(splits)
        if isinstance(Mu, list):
            if len(Mu) != len(splits):
                raise IndexError("Mu has {0:d} elements which does not match the number of splits ({1:d}).".format(len(Mu), len(splits)))
        else:
            Mu = [Mu] * len(splits)
        for split_index, split_dir in splits:
            split = Store(split_dir)
            ROMs(module, name, split, source_gp_name, Mu[split_index], Mx[split_index], optimizer_options, rbf_parameters)
    elif not isinstance(store, Fold):
        start_time = time.time()
        K_range = range(store.meta['K'])
        if K_range:
            for k in K_range:
                ROMs(module, name, Fold(store, k, M=Mx), source_gp_name, Mu, Mx, optimizer_options, rbf_parameters)
                print("Fold", k, "has finished")
        else:
            raise FileNotFoundError('Cannot construct a GP in a Store ({0:s}) which is not a Fold.'.format(store.dir))
        elapsed_mins = (time.time() - start_time) / 60
        print(store.dir.name, "has finished in {:.2f} minutes.".format(elapsed_mins))
    else:
        module.value.ROM.from_GP(fold=store, name=name, source_gp_name=source_gp_name, optimizer_options=optimizer_options, Mu=Mu,
                                 rbf_parameters=rbf_parameters)


# noinspection PyProtectedMember
def collect(store: Store, model_name: str, parameters: base.Model.Parameters, is_split: bool = True) -> Sequence[Path]:
    """Collect the Parameters of a Model.

        Args:
            store: The Store containing the global dataset to be analyzed.
            model_name: The name of the Model where the results are being collected.
            parameters: An example of the Model parameters that need to be collected.
            is_split: True or False, whether splits have been used in the Model.
        Returns: The split directories collected.
        """
    parameters = parameters._asdict()
    final_parameters = parameters.copy()
    if is_split:
        final_destination = store.dir / model_name
        final_destination.mkdir(mode=0o777, parents=True, exist_ok=True)
        splits = store.splits
    else:
        final_destination = None
        splits = [(None, store.dir)]
    for param in parameters.keys():
        for split in splits:
            split_store = Store(split[-1])
            K = split_store.meta['K']
            destination = split_store.dir / model_name
            destination.mkdir(mode=0o777, parents=True, exist_ok=True)
            for k in range(K):
                fold = Fold(split_store, k)
                source = (fold.dir / model_name) / (param + ".csv")
                if param == "Theta":
                    result = Frame(source, **base.Model.CSV_PARAMETERS).df
                else:
                    result = Frame(source, **base.Model.CSV_PARAMETERS).df.tail(1)
                result.insert(0, "Fold", full(result.shape[0], k), True)
                if k == 0:
                    parameters[param] = result.copy(deep=True)
                else:
                    parameters[param] = concat([parameters[param], result.copy(deep=True)], axis=0, ignore_index=True)
            frame = Frame(destination / (param + ".csv"), parameters[param])
            if is_split:
                result = frame.df
                result.insert(0, "Split", full(result.shape[0], split[0]), True)
                if split[0] == 0:
                    final_parameters[param] = result.copy(deep=True)
                else:
                    final_parameters[param] = concat([final_parameters[param], result.copy(deep=True)], axis=0, ignore_index=True)
        # noinspection PyUnusedLocal
        frame = Frame(final_destination / (param + ".csv"), final_parameters[param]) if is_split else None
    return splits


def collect_tests(store: Store, model_name: str, is_split: bool = True) -> Sequence[Path]:
    """Service routine to instantiate the collection of test results.

        Args:
            store: The Store containing the global dataset to be analyzed.
            model_name: The name of the model where the results are being collected.
            is_split: True or False, whether splits have been used in the model.
        Returns: The split directories collected.
    """
    final_frame = frame = None
    if is_split:
        final_destination = store.dir / model_name
        final_destination.mkdir(mode=0o777, parents=True, exist_ok=True)
        split_dirs = store.splits
    else:
        final_destination = None
        split_dirs = [store.dir]
    for split_dir in split_dirs:
        split_store = Store(split_dir)
        K = split_store.meta['K']
        destination = split_store.dir / model_name
        destination.mkdir(mode=0o777, parents=True, exist_ok=True)
        for k in range(K):
            fold = Fold(split_store, k)
            source = (fold.dir / model_name) / "__test__.csv"
            result = Frame(source).df
            result.insert(0, "Fold", full(result.shape[0], k), True)
            std = result.iloc[:, -1]
            mean = result.iloc[:, -2]
            out = result.iloc[:, -3]
            result.iloc[:, -3] = (out - mean) / std
            if k == 0:
                frame = Frame(destination / "__test__.csv", result.copy(deep=True))
            else:
                frame.df = concat([frame.df, result.copy(deep=True)], axis=0, ignore_index=False)
        frame.write()
        if is_split:
            split_index = int(split_dir.suffix[1:])
            result = frame.df
            rep = dict([(result['Predictive Mean'].columns[0], "Output")])
            result.rename(columns=rep, level=1, inplace=True)
            result.insert(0, "Split", full(result.shape[0], split_index), True)
            result = result.reset_index()
            if split_index == 0:
                final_frame = Frame(final_destination / "__test__.csv", result.copy(deep=True))
            else:
                final_frame.df = concat([final_frame.df, result.copy(deep=True)], axis=0, ignore_index=True)
    if is_split:
        final_frame.write()
    return split_dirs


def collect_GPs(store: Store, model_name: str, test: bool, sobol: bool, is_split: bool = True, kernelTypeIdentifier: str = "gpy_.ExponentialQuadratic"
                ) -> Sequence[Path]:
    """ Collect all the Parameters associated with a GP/Sobol calculation.

    Args:
        store: The Store containing the global dataset to be analyzed.
        model_name: The name of the Model where the results are being collected.
        test: Whether to collect tests or not.
        sobol: Whether to Sobol indices or not.
        kernelTypeIdentifier: The KernelTypeIdentifier for the GP.
        is_split: True or False, whether splits have been used in the Model.
    Returns: The split directories collected.

    """
    collect(store, model_name, base.GP.DEFAULT_PARAMETERS, is_split)
    if test:
        collect_tests(store, model_name, is_split)
    if sobol:
        collect(store, model_name + "\\Sobol", base.Sobol.DEFAULT_PARAMETERS, is_split)
    return collect(store, model_name + "\\Kernel", base.Kernel.TypeFromIdentifier(kernelTypeIdentifier).DEFAULT_PARAMETERS, is_split)


def rotate_inputs(gb_path: Union[str, Path], X_stand: NP.Matrix) -> NP.Matrix:
    """ Rotates the standardized inputs by theta to produce the rotated inputs that can be used when predicting with a ROM.

    Args:
        gb_path: Path to a model.GaussianBundle. The extension of this filename is the number of input dimensions M.
            An extension of 0 or a missing extension means full order, taking M from the training data.
        X_stand: The standardized input values - an (N,M) numpy array, consisting of N test inputs, each of dimension M.
    Returns:
        U: The rotated inputs that can be used for predicting using a ROM - a numpy array of dimensions (N x Mu).
    """
    gb_path = Path(gb_path)
    Mu = int(gb_path.suffix[1:])
    fold_dir = gb_path.parent
    sobol = fold_dir / "ROM.optimized" / "sobol"
    theta_T = transpose(Frame(sobol / "Theta.csv", csv_parameters={'header': [0]}).df.values)
    k = int(fold_dir.suffix[1:])
    M = Fold(fold_dir.parent, k, Mu).M +1
    if 0 < Mu < M:
        U = X_stand @ theta_T[:, 0:Mu]
    else:
        U = X_stand @ theta_T
    return U

