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

""" **Functionality for linear regression** """

from __future__ import annotations

import numpy as np

from romcomma.base.definitions import *
import scipy.stats
from romcomma.data.storage import Frame, Repository, Fold
from romcomma.user import functions
import shutil
import sys


def gls(X: NP.Matrix, y: NP.Matrix, cov_y: NP.Matrix, is_through_origin: bool = False) -> Tuple[TF.Vector, TF.Matrix]:
    """ `Generalized Least Squares <https://en.wikipedia.org/wiki/Generalized_least_squares>`_ linear regression.

    Args:
        X: An (N,M) matrix of regression variables
        y: An (N,1) vector of observations.
        cov_y: The (N,N) covariance matrix of observations ``y``.
        is_through_origin: True to constrain to ``y(0)=0``

    Returns: A pair consisting of the (M+1,1) -- or (M,1) if ``is_through_origin`` -- regression coefficients and their covariance matrix,
        where the intercept is the first regression coefficient -- or absent if ``is_through_origin``.

    """
    if not is_through_origin:
        X = tf.pad(X, ((0, 0), (0, 1)), mode='constant', constant_values=1.0)
    cov_cho = tf.linalg.cholesky(cov_y)
    precision_cho_X = tf.linalg.triangular_solve(cov_cho, X)
    precision_cho_y = tf.linalg.triangular_solve(cov_cho, y)
    cov_beta = tf.linalg.cholesky(tf.einsum('NM, Nm -> Mm', precision_cho_X, precision_cho_X))
    cov_beta = tf.linalg.triangular_solve(cov_beta, tf.eye(X.shape[-1]))
    cov_beta = tf.einsum('NM, Nm -> Mm', cov_beta, cov_beta)
    beta = tf.einsum('Mm, NM, Nl -> ml', cov_beta, precision_cho_X, precision_cho_y)
    return beta, cov_beta
