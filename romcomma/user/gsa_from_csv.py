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

""" Create a Repository by sampling a test function, adding noise, and optionally rotating the input basis """
from __future__ import annotations
import argparse
import tarfile
import os

from romcomma.base.definitions import *
from romcomma import run, data

K: int = 2     #: The number of folds. A positive value includes an extra fold(K) trained on all the data, a negative value does not.
NOISE_MAGNITUDE: float | None = None   #: The noise-to-signal ratio, which is equal to the StdDev of the noise added to the normalised function output.
IS_NOISE_COVARIANT: bool = False   #: Whether the Gaussian noise applied to the outputs is statistically independent between outputs.
IS_NOISE_VARIANCE_RANDOM: bool = False    #: Whether the noise variance is stochastic or fixed.
ROTATION: NP.MatrixLike = None    #: Rotation applied to the input basis after the function vector has been sampled.
IS_COVARIANT: bool | None = False   #: Whether the GPR likelihood is covariant.
IS_ISOTROPIC: bool | None = False    #: Whether the GPR kernel is isotropic.
KINDS: List[run.summarised.GSA.Kind] = run.summarised.GSA.ALL_KINDS    #: A list of the kinds of GSA to do.
IS_ERROR_CALCULATED: bool = True
IS_T_PARTIAL: bool = False


def run_gsa_from_function(root: Path, device: str, ignore_exceptions: bool):
    with run.context.Environment('Test', device=device):
        KIND_NAMES = [kind.name.lower() for kind in KINDS]
        repo = data.storage.Repository.from_csv(root, root.with_name(f'{root.name}.csv'), meta={'origin': {'csv': root.name}})
        if NOISE_MAGNITUDE is not None:
            noise_variance = run.sample.GaussianNoise.Variance(repo.L, NOISE_MAGNITUDE, IS_NOISE_COVARIANT, IS_NOISE_VARIANCE_RANDOM)
            run.sample.GaussianNoise(repo.N, noise_variance)(repo)
            repo.meta['origin'].update({'noise': noise_variance.meta})
            repo.write_meta()

        repo = data.storage.Repository.from_csv(root, root.with_name(f'{root.name}.csv')).into_K_folds(K, shuffle_before_folding=False)
        with context.Timer(f'M={M}, N={N}', is_inline=False):
            if READ:
                repo = sample.Function(root, DOE, FUNCTION_VECTOR, N, M, noise_variance, str(ext), False).repo
                models = [path.name for path in repo.folder.glob('gpr.*')]
            else:
                repo = (sample.Function(root, DOE, FUNCTION_VECTOR, N, M, noise_variance, str(ext), True)
                        .into_K_folds(K).rotate_folds(rotation).repo)
                models = summarised.gpr(name='gpr', repo=repo, is_read=IS_READ, is_covariant=IS_COVARIANT, is_isotropic=IS_ISOTROPIC,
                                        ignore_exceptions=ignore_exceptions)
            results.Collect({'test': {'header': [0, 1]}, 'test_summary': {'header': [0, 1], 'index_col': 0}},
                            {repo.folder / model: {'model': model} for model in models},
                            ignore_exceptions).from_folders(repo.folder / 'gpr', True)
            results.Collect({'variance': {}, 'log_marginal': {}},
                            {f'{repo.folder / model}/likelihood': {'model': model} for model in models},
                            ignore_exceptions).from_folders((repo.folder / 'gpr') / 'likelihood', True)
            results.Collect({'variance': {}, 'lengthscales': {}},
                            {f'{repo.folder / model}/kernel': {'model': model} for model in models},
                            ignore_exceptions).from_folders((repo.folder / 'gpr') / 'kernel', True)
            summarised.gsa('gpr', repo, is_covariant=IS_COVARIANT, is_isotropic=False, kinds=KINDS,
                           is_error_calculated=IS_ERROR_CALCULATED, ignore_exceptions=ignore_exceptions, is_T_partial=IS_T_PARTIAL)
            results.Collect({'S': {}, 'V': {}} | ({'T': {}, 'W': {}} if IS_ERROR_CALCULATED else {}),
                            {f'{repo.folder / model}/gsa/{kind_name}': {'model': model, 'kind': kind_name}
                             for kind_name in KIND_NAMES for model in models},
                            ignore_exceptions).from_folders((repo.folder / 'gsa'), True)

if __name__ == '__main__':
    # Gets the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", action='store_true', help="Flag to run data generation and storage.")
    parser.add_argument("-r", "--gpr", action='store_true', help="Flag to run Gaussian process regression.")
    parser.add_argument("-s", "--gsa", action='store_true', help="Flag to run global sensitivity analysis.")
    parser.add_argument("-g", "--gpu", action='store_true', help="Flag to run on a GPU instead of CPU.")
    parser.add_argument("-i", "--ignore", action='store_true', help="Flag to ignore exceptions.")
    parser.add_argument("-t", "--tar", help="Outputs a .tar.gz file to path", type=str)
    parser.add_argument("root", help="The path of the root folder to house all data repositories.", type=str)
    args = parser.parse_args()  # Convert arguments to dictionary

    # Runs the code
    print(f'Root path is {args.root}')
    run_gsa_from_function(root=Path(args.root), device='GPU' if args.gpu else 'CPU', )

    # Tarring outputs
    if args.tar:
        # Gets the tar path and ensure we have a directory
        tar_path = Path(args.tar)
        tar_path.parents[0].mkdir(parents=True, exist_ok=True)
        # Puts everything in the ROOT folder inside the tar file
        with tarfile.open(tar_path, "w:gz") as tar:
            for item in os.listdir(args.root):
                tar.add(Path(args.root, item), arcname=item)




