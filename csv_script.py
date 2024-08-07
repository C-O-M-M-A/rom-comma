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


""" Benchmark GPR and GSA for known test functions. """


from __future__ import annotations

import numpy as np

from romcomma.base.definitions import *
from romcomma import user, data
from romcomma.gpr import kernels
import argparse
import tarfile
import os

#: Parameters for repository generation.
K: int = 20  #: The number of Folds in a new repository.
INPUT_AXIS_PERMUTATIONS: Dict[str, List[int] | None] = {'': None}   # , '.0': [5, 2, 3, 1, 4, 6, 0], '.1': [5, 3, 2, 4, 6, 1, 0], '.2': [5, 2, 1, 3, 4, 0, 6], '.3': [5, 2, 6, 1, 3, 0, 6]}
                                                        # '.520': [5, 2, 0, 6, 3, 4, 1]}   #: A Dict of the form {path_suffix, input_axis_permutation}.
#: Parameters to run Gaussian Process Regression.
IS_GPR_READ: bool | None = False  #: Whether to read the GPR model from file.
IS_GPR_COVARIANT: bool | None = False  #: Whether the GPR likelihood is covariant.
IS_GPR_ISOTROPIC: bool | None = False  #: Whether the GPR kernel is isotropic.
#: Parameters to run Global Sensitivity Analysis.
GSA_KINDS: List[user.run.GSA.Kind] = user.run.GSA.ALL_KINDS  #: A list of the kinds of GSA to do.
IS_GSA_ERROR_CALCULATED: bool = True  #: Whether to calculate the GSA standard error.
IS_GSA_ERROR_PARTIAL: bool = False  #: Whether the calculated the GSA standard error is partial.


def run(root: str | Path, csv: str | Path, gpr: bool = False, gsa: bool = False, ignore_exceptions: bool = True, use_gpu: bool = False, k: int | None = None,
        normalization: str | None = None, unnormalized: bool = False, likelihood_variance: float | None = None,
        rbf_variance: float | None = None, rbf_lengthscale: float | None = None, coregional_variance: float | None = None) -> Path:
    """ Run benchmark data generation and/or Gaussian Process Regression and/or Global Sensitivity Analysis, and collect the results.

    Args:
        root: The root folder.
        csv: The csv to read.
        gpr: Whether to perform gpr.
        gsa: Whether to perform gsa.
        ignore_exceptions: Whether to continue in spite of errors in GPR or GSA, missing files, etc.
        use_gpu: Whether to use GPU or CPU.
        k: The number of folds.
        normalization: An optional csv file to use for normalization.
        unnormalized: False unless data has already been normalized.
        likelihood_variance: Initial guess for the likelihood variance hyperparameters.
        rbf_variance: Initial guess for radial basis function variance hyperparameters.
        rbf_lengthscale: Initial guess for rbf lengthscale hyperparameters.
        coregional_variance: Initial guess for coregional variance hyperparameter.
    Returns: The root path written to.
    """
    with user.contexts.Environment('Test', device='/GPU' if use_gpu else '/CPU'):
        KIND_NAMES = [kind.name.lower() for kind in GSA_KINDS]
        gprs, gsas = {}, {}
        k = K if k is None else k
        rbf_folder = root / 'rbf'
        if rbf_variance is None:
            kernel_parameters = None if rbf_lengthscale is None else kernels.RBF.Data(folder=rbf_folder, lengthscales=np.atleast_2d(rbf_lengthscale))
        elif rbf_lengthscale is None:
            kernel_parameters = kernels.RBF.Data(folder=rbf_folder, variance=np.atleast_2d(rbf_variance))
        else:
            kernel_parameters = kernels.RBF.Data(folder=rbf_folder, lengthscales=np.atleast_2d(rbf_lengthscale), variance=np.atleast_2d(rbf_variance))
        for ext, permutation in INPUT_AXIS_PERMUTATIONS.items():
            repo_folder = root if len(INPUT_AXIS_PERMUTATIONS) == 1 else (root / root.name).with_suffix(root.suffix + ext)
            with user.contexts.Timer(f'ext={ext}', is_inline=False):
                if gpr:
                    # Get data from csv then run GPR.
                    repo = (data.storage.Repository.from_csv(repo_folder, csv)
                            .into_K_folds(k, normalization=normalization, is_normalization_applicable=not unnormalized)
                            .rotate_folds(user.sample.permute_axes(permutation)))
                    models = user.run.gpr(name='gpr', repo=repo, is_read=IS_GPR_READ, is_covariant=IS_GPR_COVARIANT,
                                          is_isotropic=IS_GPR_ISOTROPIC, ignore_exceptions=ignore_exceptions,
                                          kernel_parameters=kernel_parameters, likelihood_variance=likelihood_variance)
                else:
                    # Collect stored GPR models.
                    repo = data.storage.Repository(repo_folder)
                    models = [path.name for path in repo.folder.glob('gpr.*')]

                # Collect GPR results from GPR models.
                user.results.Collect({'test': {'header': [0, 1]}, 'test_summary': {'header': [0, 1]}},
                                     {repo.folder / model: {'model': model} for model in models},
                                     True).from_folders(repo.folder / 'gpr', True)
                user.results.Collect({'variance': {}, 'log_marginal': {}},
                                     {f'{repo.folder / model}/likelihood': {'model': model} for model in models},
                                     True).from_folders((repo.folder / 'gpr') / 'likelihood', True)
                user.results.Collect({'variance': {}, 'lengthscales': {}},
                                     {f'{repo.folder / model}/kernel': {'model': model} for model in models},
                                     True).from_folders((repo.folder / 'gpr') / 'kernel', True)
                gprs |= {f'{repo.folder}/gpr': {'ext': ext}}
                # Run GSA and collect results, or just collect results.
                if gsa:
                    user.run.gsa('gpr', repo, is_covariant=IS_GPR_COVARIANT, is_isotropic=False, kinds=GSA_KINDS,
                                 is_error_calculated=IS_GSA_ERROR_CALCULATED, ignore_exceptions=ignore_exceptions, is_T_partial=IS_GSA_ERROR_PARTIAL)
                user.results.Collect({'S': {}, 'V': {}} | ({'T': {}, 'W': {}} if IS_GSA_ERROR_CALCULATED else {}),
                                     {f'{repo.folder / model}/gsa/{kind_name}': {'model': model, 'kind': kind_name}
                                      for kind_name in KIND_NAMES for model in models},
                                     True).from_folders((repo.folder / 'gsa'), True)
                gsas |= {f'{repo.folder}/gsa': {'ext': ext}}
    kernels.RBF.Data.delete(rbf_folder)
    user.results.Collect({'test_summary': {'header': [0, 1]}}, gprs, True).from_folders(root / 'gpr', False)
    user.results.Collect({'variance': {}, 'log_marginal': {}}, {key + '/likelihood': value for key, value in gprs.items()},
    True).from_folders((root / 'gpr') / 'likelihood', False)
    user.results.Collect({'variance': {}, 'lengthscales': {}}, {key + '/kernel': value for key, value in gprs.items()},
                         True).from_folders((root / 'gpr') / 'kernel', False)
    user.results.Collect({'S': {}, 'V': {}, 'T': {}, 'W': {}}, gsas, True).from_folders((root / 'gsa'), False)
    return root


if __name__ == '__main__':
    # Get the command line arguments.
    parser = argparse.ArgumentParser(description='A program to benchmark GPR and GSA against a (vector) test function.')
    # Control Flow.
    parser.add_argument('-r', '--gpr', action='store_true', help='Flag to run Gaussian process regression.')
    parser.add_argument('-a', '--gsa', action='store_true', help='Flag to run global sensitivity analysis.')
    parser.add_argument('-i', '--ignore', action='store_true', help='Flag to ignore exceptions.')
    parser.add_argument('-u', '--unnormalized', action='store_true', help='Flag to use unnormalized data.')
    parser.add_argument('-G', '--GPU', action='store_true', help='Flag to run on a GPU instead of CPU.')
    parser.add_argument("-l", "--likelihood_variance", help="Initial guess for likelihood variance to be calibrated.", type=float)
    parser.add_argument("-s", "--rbf_lengthscale", help="Initial guess for rbf lengthscale to be calibrated.", type=float)
    parser.add_argument("-v", "--rbf_variance", help="Initial guess for the rbf variance to be calibrated.", type=float)
    parser.add_argument("-c", "--coregional_variance", help="Initial guess for coregional variance to be calibrated.", type=float)
    parser.add_argument("-K", "--folds", help="K, the number of folds for K-fold validation.", type=int)
    parser.add_argument('-k', '--proper', action='store_true', help='Flag to suppress improper fold.')
    # File locations.
    parser.add_argument('-t', '--tar', help='Outputs a .tar.gz file to path.', type=str)
    parser.add_argument('-n', '--normalization', help='A csv file to use for normalization.', type=str)
    parser.add_argument('csv', help='The path of the csv containing the data to be analysed.', type=Path)
    parser.add_argument('root', help='The path of the root folder to house all data repositories.', type=Path)
    args = parser.parse_args()  # Convert arguments to argparse.Namespace.
    # Run the code.
    k = None if args.folds is None else (-args.folds if args.proper else args.folds)
    print(f'''Root path is {run(args.root, args.csv, args.gpr, args.gsa, args.ignore, args.GPU, k, args.normalization, args.unnormalized, 
                              args.likelihood_variance, args.rbf_variance, args.rbf_lengthscale, args.coregional_variance)}.''')
    # Tar outputs
    if args.tar:
        tar = Path(args.tar)
        tar.parents[0].mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar, 'w:gz') as tar:
            for item in os.listdir(args.root):
                tar.add(Path(args.root, item), arcname=item)
