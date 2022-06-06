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

from romcomma.base.definitions import *
from romcomma.data.storage import Repository, Fold
import romcomma
import shutil

def aggregator(child_path: Union[Path, str], function_names: Sequence[str], N: int, noise_magnitude: float, random: bool, M: int = 5) -> Dict[str, Any]:
    return {'path': repo_folder(function_names, N, noise_label(noise_magnitude), random, M) / child_path, 'N': N, 'noise': noise_magnitude}


def aggregate(aggregators: Dict[str, Sequence[Dict[str, Any]]], dst: Union[Path, str], ignore_missing: bool=False, **kwargs):
    """ Aggregate csv files over aggregators.

    Args:
        aggregators: A Dict of aggregators, keyed by csv filename. An aggregator is a List of Dicts containing source path ['path']
            and {key: value} to insert column 'key' and populate it with 'value' in path/csv.
        dst: The destination path, to which csv files listed as the keys in aggregators.
        **kwargs: Write options passed directly to pd.Dataframe.to_csv(). Overridable defaults are {'index': False, 'float_format':'%.6f'}
    """
    dst = Path(dst)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(mode=0o777, parents=True, exist_ok=False)
    kwargs = {'index': False, 'float_format': '%.6f'} | kwargs
    for csv, aggregator in aggregators.items():
        is_initial = True
        results = None
        for file in aggregator:
            filepath = Path(file.pop('path'))/csv
            if filepath.exists() or not ignore_missing:
                result = pd.read_csv(filepath)
                for key, value in file.items():
                    result.insert(0, key, np.full(result.shape[0], value), True)
                if is_initial:
                    results = result.copy(deep=True)
                    is_initial = False
                else:
                    results = pd.concat([results, result.copy(deep=True)], axis=0, ignore_index=True)
        results.to_csv(dst/csv, index=False)


if __name__ == '__main__':
    # BASE_PATH = BASE_ROOT
    # for gsa in ('first_order.d', 'first_order.p.d'):
    #     with run.Context(f'GSA.{gsa}', float='float64', device='CPU'):  #
    #         child_path = Path('initial.i.a\\gsa') / gsa
    #         csvs = ['S.csv', 'T.csv']
    #         aggregators = {csv: [] for csv in csvs}
    #         for test in range(6):
    #             for csv in csvs:
    #                 aggregators[csv].append({'path': BASE_PATH/f'9.{test}'/child_path, 'test': test})
    #         aggregate(aggregators, dst=BASE_PATH/child_path, ignore_missing=True)
