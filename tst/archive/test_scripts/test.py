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

""" Run this module first thing, to test your installation of romcomma.

**Contents**:
    **predict**: Prediction using a GaussianBundle.

    **test_input**: A rudimentary test input, for installation testing.
"""

from romcomma import model, data
import numpy as np
from romcomma.typing_ import NP, Union, Tuple
from pathlib import Path
from romcomma import distribution


def predict(gb_path: Union[str, Path], inputs: NP.Array) -> Tuple[NP.Vector, NP.Vector]:
    """ Prediction using a GaussianBundle.

    Args:
        gb_path: Path to a model.GaussianBundle. The extension of this filename is the number of input dimensions M.
            An extension of 0 or a missing extension means full order, taking M from the training data.
        inputs: An (N,M) numpy array, consisting of N test inputs, each of dimension M.
    Returns: A pair (predictive_mean, predictive_std) of (N,1) numpy arrays.
    """
    gb_path = Path(gb_path)
    gb_name = gb_path.name
    Xs_taken = int(gb_path.suffix[1:])
    fold_dir = gb_path.parent
    k = int(fold_dir.suffix[1:])
    fold = data.Fold(fold_dir.parent, k, Xs_taken)
    if Xs_taken != inputs.shape[1]:
        raise IndexError("The gp you have chosen uses {0:d} dimensional inputs".format(fold.M))
    gb = model.gpy_.GaussianBundle(fold, gb_name, reset_log_likelihood=False)
    return gb.predict(inputs)


def test_input() -> NP.Vector:
    return np.transpose(np.atleast_2d(np.linspace(-1.0, 1.0, num=3)))


def exploratory_xi(N_explore: int, xi_len: int) -> NP.Matrix:
    dist = distribution.Multivariate.Independent(xi_len + 1, distribution.Univariate('uniform', loc=-1, scale=2))
    result = dist.sample(N_explore, distribution.SampleDesign.LATIN_HYPERCUBE)
    norm = np.sqrt(np.sum(result ** 2, axis=1).reshape((N_explore, 1)))
    return result[:, 1:] / norm


if __name__ == '__main__':
    bum = exploratory_xi(2, 5)
    sgn = np.sign(np.diag(bum))
    print(bum, sgn)

""" Test Installation. """
"""
if __name__ == '__main__':
    mean_prediction, std_prediction = predict("resources\\UKCCSRC\\UoS\\Results\\split.0\\fold.0\\ARD.1", test_input())
    print("input\n", test_input(), "\n")
    print("predictive mean\n", mean_prediction, "\n")
    print("predictive std\n", std_prediction, "\n")
    print("INSTALLATION TEST SUCCEEDED!!!")
"""