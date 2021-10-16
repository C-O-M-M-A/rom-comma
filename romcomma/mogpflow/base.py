#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2021 Robert A. Milton. All rights reserved.
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

""" Contains extensions to gpflow.base."""

from __future__ import annotations

from gpflow import Parameter
from gpflow.utilities import positive
from gpflow.models.util import data_input_to_tensor
import tensorflow as tf

class Covariance:
    """ A non-diagonal Covariance Matrix."""

    DEFAULT_CHOLESKY_DIAGONAL_LOWER_BOUND = 1e-3

    @property
    def shape(self):
        """ Returns (L,L), which is the shape of self.value and self.cholesky."""
        return self._shape

    @property
    def cholesky(self):
        """ The (lower triangular) Cholesky decomposition of the covariance matrix."""
        self.refresh()
        return self._cholesky

    @property
    def value(self):
        """ The covariance matrix, shape (L,L)."""
        self.refresh()
        return self._value

    @property
    def variance(self):
        """ The covariance matrix, shape (L,1,L,1) ready to broadcast."""
        self.refresh()
        return self._variance

    def refresh(self):
        """ Refresh properties if the Parameters _cholesky_diagonal and _cholesky_lower_triangle have changed."""
        if tf.reduce_all(tf.equal(self._cholesky_lower_triangle, self._cholesky_lower_triangle_stale)):
            if tf.reduce_all(tf.equal(self._cholesky_diagonal, self._cholesky_diagonal_stale)):
                return False
        else:
            self._cholesky = \
                tf.RaggedTensor.from_row_lengths(self._cholesky_lower_triangle, self._row_lengths).to_tensor(default_value=0, shape=self._shape)
            self._cholesky_lower_triangle_stale = tf.identity(self._cholesky_lower_triangle)
        self._cholesky = tf.linalg.set_diag(self._cholesky, self._cholesky_diagonal)
        self._value = tf.matmul(self._cholesky, self._cholesky, transpose_b=True)
        self._variance = tf.reshape(self._value, self._variance_shape)
        self._cholesky_diagonal_stale = tf.identity(self._cholesky_diagonal)
        return True

    def __init__(self, value, name: str = 'Covariance', cholesky_diagonal_lower_bound: float = DEFAULT_CHOLESKY_DIAGONAL_LOWER_BOUND):
        """ Construct a non-diagonal covariance matrix. Mutable only through it's properties cholesky_diagonal and cholesky_lower_triangle.

        Args:
            value: A symmetric, positive definite matrix, expressed in tensorflow or numpy.
            cholesky_diagonal_lower_bound: Lower bound on the diagonal of the Cholesky decomposition.
        """
        value = data_input_to_tensor(value)
        self._shape = (value.shape[-1], value.shape[-1])
        self._variance_shape = (value.shape[-1], 1, value.shape[-1], 1)
        if value.shape != self._shape:
            raise ValueError('Covariance must have shape (L,L).')

        cholesky = tf.linalg.cholesky(value)

        self._cholesky_diagonal = tf.linalg.diag_part(cholesky)
        if min(self._cholesky_diagonal) <= cholesky_diagonal_lower_bound:
            raise ValueError(f'The Cholesky diagonal of {name} must be strictly greater than {cholesky_diagonal_lower_bound}.')
        self._cholesky_diagonal = Parameter(self._cholesky_diagonal, transform=positive(lower=cholesky_diagonal_lower_bound),
                                               name=name+'.cholesky_diagonal')

        mask = sum([list(range(i * self._shape[0], i * (self._shape[0] + 1))) for i in range(1, self._shape[0])], start=[])
        self._cholesky_lower_triangle = Parameter(tf.gather(tf.reshape(cholesky, [-1]), mask), name=name+'.cholesky_lower_triangle')

        self._row_lengths = tuple(range(self._shape[0]))
        self._cholesky_diagonal_stale, self._cholesky_lower_triangle_stale = self._cholesky_diagonal + 1, self._cholesky_lower_triangle + 1
