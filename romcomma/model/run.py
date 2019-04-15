# BSD 3-Clause License
#
# Copyright (c) 2019, Robert A. Milton
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Contains routines for running models.

**Contents**:
    **Module**: The class that defines the module names bounding them to the scripts.

    **_validate_lengthscale_trainer**: Function to ensure the variable lengthscale_trainer is of correct size.

    **gb**: Function to instantiate a GaussianBundle.

    **_gbs**: Function that runs Gaussian Bundles across the Folds of a store.

    **gbs**: Function to instantiate, optimize and test GaussianBundles on a store.

    **ROM_fold**: Service routine to instantiate a Reduced Order Model on a Fold.

    **ROM_store**: Service routine to instantiate a Reduced Order Model across the Folds of a store.

    **collect**: Service routine to instantiate the collection of new parameters.

    **collect_tests**: Service routine to instantiate the collection of test results.
"""

from romcomma.typing_ import NP, Optional, Sequence, Tuple, NamedTuple
from romcomma.data import Store, Fold, Frame
from romcomma.model.base import GaussianBundle, Model, ROM, SemiNorm, Sobol
from numpy import atleast_1d, atleast_2d, full, broadcast_to
from pandas import concat
from enum import Enum
from pathlib import Path
from romcomma.model import mygpy, myscipy
import time


class Module(Enum):
    """ Defines an enumeration which has 2 members, each with a name which is now bound to a module.

    """
    MYGPY = mygpy
    MYSCIPY = myscipy


def _validate_lengthscale_trainer(lengthscale_trainer: NP.ArrayLike, L: int) -> Tuple[Optional[NP.Vector], int]:
    """ Service routine to ensure the variable lengthscale_trainer is of correct size.

        Args:
            lengthscale_trainer: A list of outputs to train on. A scalar l less than L is interpreted as the first l outputs.
            L: The number of outputs.
        Raises:
            IndexError: If lengthscale_trainer is not a CoVector or scalar.
            ValueError: If lengthscale_trainer is outside of the range of L.
        Returns:
            A Tuple containing a Vector of lengthscales (or None), together with its len.
        """
    if lengthscale_trainer is None:
        return None, 1
    lengthscale_trainer = atleast_2d(lengthscale_trainer)
    rangeL = list(range(L))
    if lengthscale_trainer.shape[0] > 1:
        raise IndexError("parameters.lengthscale_trainer must be a CoVector or scalar.")
    elif lengthscale_trainer.shape[1] == 1 and lengthscale_trainer[0, 0] not in rangeL:
        lengthscale_trainer = atleast_2d(rangeL)
    else:
        for i in range(1, lengthscale_trainer.shape[1]):
            if lengthscale_trainer[0, i] not in rangeL:
                raise ValueError("lengthscale_trainer[0, {i:d}] not in range(L)".format(i=i))
    return lengthscale_trainer, lengthscale_trainer.shape[1]


# noinspection PyProtectedMember
def gb(module: Module, fold: Fold, gb_name: str, parameters: Optional[GaussianBundle.Parameters],
       lengthscale: NP.ArrayLike, ard: bool) -> GaussianBundle:
    """ Service routine to instantiate a GaussianBundle

    Args:
        module: Sets the implementation to either Module.MYGPY or Module.MYSCIPY.
        fold: Fold is a data.Store equipped with a test frame backed by ``__test__.csv``.
        gb_name: The name of the GB.
        parameters: An optional ``GaussianBundle.Parameters``. If None, parameters are read from fold.dir/gb_name.
            parameters.kernel must contain the kernel type, a subclass of Kernel. parameters.log_likelihood is not used.
        lengthscale: A scalar or 1d array of initial lengthscales for an RBF/ARD kernel.
        ard: False for RBF Kernel.
    Returns: The GaussianBundle initiated.
    """
    L, M = fold.meta['data']['L'], fold.M
    if parameters is not None:
        lengthscale = atleast_1d(lengthscale)
        if ard and lengthscale.shape != (M,):
            lengthscale = full(M, lengthscale[0], dtype=float)
        kernel_params = parameters.kernel.Parameters(f=parameters.f, lengthscale=lengthscale, ard=ard,
                                                     lengthscale_trainer=parameters.lengthscale_trainer)
        parameters = parameters._replace(kernel=[[kernel_params]], log_likelihood=None)
    return module.value.GaussianBundle(fold, gb_name, parameters, True)


def _gbs(module: Module, store: Store, gb_name_stem: str, optimize: bool, test: bool, sobol: bool, parameters: Optional[GaussianBundle.Parameters],
         lengthscale_trainer: NP.ArrayLike, lengthscale: NP.MatrixLike, ard: bool, Xs_taken: int = -1):
    """ Service routine to run gbs across the Folds of store, optimized on the output(s) indexed by lengthscale_trainer.

    Args:
        module: Sets the implementation to either Module.MYGPY or Module.MYSCIPY.
        store: Store is the dir containing the global dataset to be analyzed.
        gb_name_stem: The stem gb_name, to which the lengthscale_trainer will appended as an extension.
        optimize: Scalar, whether to optimize each GaussianBundle
        test: Scalar, whether to test each GaussianBundle
        sobol: Scalar, whether to calculate sobol' indices for each GaussianBundle
        lengthscale_trainer: A list of outputs to train on. A scalar l less than L is interpreted as the first l outputs.
        parameters: An optional ``GaussianBundle.Parameters``. If None, parameters are read from fold.dir/gb_name.
            parameters.kernel must contain the kernel type, a subclass of Kernel. parameters.log_likelihood is not used.
        lengthscale: A a CoVector, scalar or (Ls, M) matrix of initial lengthscales for an RBF/ARD kernels.
        ard: Scalar, False for RBF.
        Xs_taken: The number of X columns used. If not 0 < _Xs_taken < self.M, all columns are used.
    Raises:
        TypeError: If parameters.e_floor is not a float.
        IndexError: Unless parameters.e.shape == parameters.f.shape in (1,1), (1,len(lengthscale_trainer)).
    """
    L, M, K = store.meta['data']['L'], store.meta['data']['M'], store.meta['K']
    if 0 < Xs_taken < M:
        # noinspection PyUnusedLocal
        M = Xs_taken
    lengthscale_trainer, Ls = _validate_lengthscale_trainer(lengthscale_trainer, L)

    def _validate_lengthscale() -> NP.Matrix:
        _lengthscale = atleast_2d(lengthscale)
        if _lengthscale.shape[0] == 1:
            _lengthscale = broadcast_to(_lengthscale, (Ls, _lengthscale.shape[1])).copy()
        elif _lengthscale.shape[0] != Ls:
            raise IndexError("parameters.lengthscale must be a CoVector or scalar, or (Ls, M) matrix, not a {0} matrix".format(_lengthscale.shape))
        return _lengthscale

    lengthscale = _validate_lengthscale()

    # noinspection PyProtectedMember
    def _validate_parameters() -> GaussianBundle.Parameters:
        _parameters = parameters._replace(lengthscale_trainer=lengthscale_trainer, e=atleast_2d(parameters.e), f=atleast_2d(parameters.f),
                                          log_likelihood=full((1, Ls), 0, dtype=float))
        if not isinstance(_parameters.e_floor, float):
            raise TypeError("parameters.e_floor must be a single float.")
        if lengthscale_trainer is None:
            if not (_parameters.e.shape == _parameters.f.shape == (1, 1)):
                raise IndexError("Bad shapes parameters.e={0}, parameters.f={1}. Should both be shaped (1,1) when lengthscale_trainer is None.")
        else:
            if _parameters.e.shape == _parameters.f.shape in ((1, 1), (Ls, 1), (1, L), (Ls, L)):
                _parameters = _parameters._replace(e=broadcast_to(_parameters.e, (Ls, L)).copy(), f=broadcast_to(_parameters.f, (Ls, L)).copy())
            else:
                raise IndexError("Bad shapes parameters.e={0}, parameters.f={1}. Should be same shape, either (1,1) or (Ls, 1) or (1, L) or (Ls, L).")
        return _parameters

    if parameters is not None:
        parameters = _validate_parameters()

    def _optimize_test_and_sobol(gbu):
        if optimize:
            gbu.optimize()
        if test:
            gbu.test()
        if sobol:
            module.value.Sobol(gbu)
            print("Fold", k, "has finished")

    if lengthscale_trainer is None:
        for k in range(K):
            fold = Fold(store, k, Xs_taken)
            _optimize_test_and_sobol(gb(module, fold, Model.name(gb_name_stem), parameters, lengthscale, ard))
    else:
        for k in range(K):
            fold = Fold(store, k, Xs_taken)
            for i in range(Ls):
                lt = lengthscale_trainer[0, i]
                params = None if parameters is None else GaussianBundle.Parameters(kernel=parameters.kernel, lengthscale_trainer=lt,
                                                                                   e_floor=parameters.e_floor, e=parameters.e[i], f=parameters.f[i],
                                                                                   log_likelihood=None)
                _optimize_test_and_sobol(gb(module, fold, Model.name(gb_name_stem, lt), params, lengthscale[i], ard))


def gbs(module: Module, store: Store, gb_name_stem: str, optimize: bool, test: bool, sobol: bool, parameters: Optional[GaussianBundle.Parameters],
        lengthscale_trainer: NP.ArrayLike, lengthscale: NP.MatrixLike, ard: bool, Xs_taken: int = -1):
    """Instantiate, optimize and test GaussianBundles on a store

    Args:
        module: Sets the implementation to either Module.MYGPY or Module.MYSCIPY.
        store: Store is the dir containing the global dataset to be analyzed.
        gb_name_stem: The stem gb_name, to which the lengthscale_trainer will appended as an extension.
        optimize: Scalar, whether to optimize each GaussianBundle
        test: Scalar, whether to test each GaussianBundle
        sobol: Scalar, whether to calculate sobol' indices for each GaussianBundle
        lengthscale_trainer: A list of outputs to train on. A scalar l less than L is interpreted as the first l outputs.
        parameters: An optional ``GaussianBundle.Parameters``. If None, parameters are read from fold.dir/gb_name.
            parameters.kernel must contain the kernel type, a subclass of Kernel. parameters.log_likelihood is not used.
        lengthscale: A scalar or 1d array of initial lengthscales for an RBF/ARD kernel.
        ard: False for RBF Kernel.
        Xs_taken: The number of X columns used. If not 0 < _Xs_taken< self.M, all columns are used.
    Raises:
        TypeError: If parameters.e_floor is not a float.
        IndexError: Unless parameters.e.shape == parameters.f.shape in (1,1), (1,len(lengthscale_trainer)).
    """
    splits = store.splits
    if splits and lengthscale_trainer == 0:
        for split_dir in splits:
            split_start = time.time()
            split = Store(split_dir)
            _gbs(module, split, gb_name_stem, optimize, test, sobol, parameters, None, lengthscale, ard, Xs_taken)
            split_time_mins = (time.time() - split_start) / 60
            print(split_dir.name, "has finished in {:.2f} minutes.".format(split_time_mins))
    else:
        _gbs(module, store, gb_name_stem, optimize, test, sobol, parameters, lengthscale_trainer, lengthscale, ard, Xs_taken)


def ROM_fold(module: Module, fold: Fold, gb_name_stem: str, parameters: Optional[GaussianBundle.Parameters], lengthscale: NP.ArrayLike,
             ard: bool, allow_rotation: bool, iterations: int, guess_identity_after_iteration: int = -1, N_search: int = 2048,
             N_optimize: int = 1, xi: Optional[NP.Array] = None, reuse_default_gb_parameters: bool = False, options: dict = Sobol.OPTIMIZER_OPTIONS)\
            -> ROM:
    """Service routine to instantiate a Reduced Order Model on a Fold.

    Args:
        module: Sets the implementation to either Module.MYGPY or Module.MYSCIPY.
        fold: Fold is a data.Store equipped with a test frame backed by ``__test__.csv``.
        gb_name_stem: The stem gb_name, to which the lengthscale_trainer will appended as an extension.
        parameters: An optional ``GaussianBundle.Parameters``. If None, parameters are read from fold.dir/gb_name.
            parameters.kernel must contain the kernel type, a subclass of Kernel. parameters.log_likelihood is not used.
        lengthscale: A scalar or 1d array of initial lengthscales for an RBF/ARD kernel.
        ard: False for RBF Kernel.
        allow_rotation: True for full rotated reduced order modelling.
            Otherwise, False for re-ordering and optimisation, then the following Args would be obsolete.
        iterations: The maximum number of top-level sobol' optimisation iterations allowed.
        guess_identity_after_iteration: After this number of sobol' iterations, theta will be guessed as the identity matrix.
        N_search: The number of random or grid search guesses used to initiate Xi optimization.
        N_optimize: The best N-Optimize Xi values from grid search are gradient-descent optimized to find an overall minimum.
        xi: Specify the initial guess to use for xi optimization. Rarely useful, so the default is None.
        reuse_default_gb_parameters: True or False whether to reuse the default parameters of the GB after each iteration.
        options: scipy.optimize.minimize(). For further details see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
    Returns: The Reduced Order Model initiated.
    """
    assert fold.L == 1
    gbu = gb(module, fold, gb_name_stem + ".0", parameters, lengthscale, ard)
    sobol = module.value.Sobol(gbu)
    result = ROM(sobol, SemiNorm.element(gbu.L, 0, 0))
    # noinspection PyArgumentList
    result.optimize(allow_rotation, iterations, guess_identity_after_iteration, N_search, N_optimize, xi,
                    reuse_default_gb_parameters, options)
    return result


def ROM_store(module: Module, store: Store, gb_name_stem: str, parameters: Optional[GaussianBundle.Parameters], lengthscale: NP.ArrayLike,
              ard: bool, allow_rotation: bool, iterations: int, guess_identity_after_iteration: int = -1, N_search: int = 2048,
              N_optimize: int = 1, xi: Optional[NP.Array] = None, reuse_default_gb_parameters: bool = False, options: dict = Sobol.OPTIMIZER_OPTIONS):
    """Service routine to instantiate a Reduced Order Model across the Folds of a store.

    Args:
        module: Sets the implementation to either Module.MYGPY or Module.MYSCIPY.
        store: Store is the dir containing the global dataset to be analyzed.
        gb_name_stem: The stem gb_name, to which the lengthscale_trainer will appended as an extension.
        parameters: An optional ``GaussianBundle.Parameters``. If None, parameters are read from fold.dir/gb_name.
            parameters.kernel must contain the kernel type, a subclass of Kernel. parameters.log_likelihood is not used.
        lengthscale: A scalar or 1d array of initial lengthscales for an RBF/ARD kernel.
        ard: False for RBF Kernel.
        allow_rotation: True for full rotated reduced order modelling.
            Otherwise, False for re-ordering and optimisation, then the following Args would be obsolete.
        iterations: The maximum number of iterations allowed.
        guess_identity_after_iteration: After this number of iterations, Theta will be guessed as the identity matrix. Default is -1.
        N_search: The number of random or grid search guesses used to initiate Xi optimization.
        N_optimize: The best N-Optimize Xi values from grid search are gradient-descent optimized to find an overall minimum.
        xi: Specify the initial guess to use for xi optimization. Rarely useful, so the default is None.
        reuse_default_gb_parameters: True or False whether to reuse the default parameters of the GB after each iteration. Default is False.
        options: scipy.optimize.minimize(). For further details see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
    Raises:
        TypeError: If parameters.e_floor is not a float.
        IndexError: Unless parameters.e.shape == parameters.f.shape in (1,1), (1,len(lengthscale_trainer)).
        UserWarning: If allow_optimisation is False, iterations will be changed to 1.
    """
    for split_dir in store.splits:
        split_start_time = time.time()
        split = Store(split_dir)
        for k in range(split.K):
            ROM_fold(module, Fold(split, k), gb_name_stem, parameters, lengthscale, ard, allow_rotation, iterations,
                     guess_identity_after_iteration, N_search, N_optimize, xi, reuse_default_gb_parameters, options)
            print("Fold", k, "has finished")
        split_sobol_time_mins = (time.time() - split_start_time) / 60
        print(split_dir.name, "has finished in {:.2f} minutes.".format(split_sobol_time_mins))


# noinspection PyProtectedMember
def collect(store: Store, model_name: str, parameters: NamedTuple, is_split: bool = True) -> Sequence[Path]:
    """Service routine to instantiate the collection of new parameters.

        Args:
            store: The Store containing the global dataset to be analyzed.
            model_name: The name of the model where the results are being collected.
            parameters: An example of the model parameters that need to be collected.
            is_split: True or False, whether splits have been used in the model.
        Returns: The split directories collected.
        """
    parameters = parameters._asdict()
    final_parameters = parameters.copy()
    if is_split:
        final_destination = store.dir / model_name
        final_destination.mkdir(mode=0o777, parents=True, exist_ok=True)
        split_dirs = store.splits
    else:
        final_destination = None
        split_dirs = [store.dir]
    for param in parameters.keys():
        for split_dir in split_dirs:
            split_store = Store(split_dir)
            K = split_store.meta['K']
            destination = split_store.dir / model_name
            destination.mkdir(mode=0o777, parents=True, exist_ok=True)
            for k in range(K):
                fold = Fold(split_store, k)
                source = (fold.dir / model_name) / (param + ".csv")
                if param == "Theta":
                    result = Frame(source, csv_parameters=Model.CSV_PARAMETERS).df
                else:
                    result = Frame(source, csv_parameters=Model.CSV_PARAMETERS).df.tail(1)
                result.insert(0, "Fold", full(result.shape[0], k), True)
                if k == 0:
                    parameters[param] = result.copy(deep=True)
                else:
                    parameters[param] = concat([parameters[param], result.copy(deep=True)], axis=0, ignore_index=True)
            frame = Frame(destination / (param + ".csv"), parameters[param])
            if is_split:
                split_index = int(split_dir.suffix[1:])
                result = frame.df
                result.insert(0, "Split", full(result.shape[0], split_index), True)
                if split_index == 0:
                    final_parameters[param] = result.copy(deep=True)
                else:
                    final_parameters[param] = concat([final_parameters[param], result.copy(deep=True)], axis=0, ignore_index=True)
        # noinspection PyUnusedLocal
        frame = Frame(final_destination / (param + ".csv"), final_parameters[param]) if is_split else None
    return split_dirs


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
