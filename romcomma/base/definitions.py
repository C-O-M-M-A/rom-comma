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

""" Type and constant definitions. """

from __future__ import annotations

from typing import *
from pathlib import Path
import numpy as np
import tensorflow as tf
import gpflow as gf
import romcomma.gpf as mf
import pandas as pd
from abc import abstractmethod


EFFECTIVELY_ZERO = 1.0E-64  #: Tolerance when testing floats for equality.


def INT() -> Type:
    """ The ``dtype`` of ``int`` in :ref:`romcomma.run.context.Environment`. """
    return gf.config.default_int()


def FLOAT() -> Type:
    """ The ``dtype`` of ``float`` in :ref:`romcomma.run.context.Environment`. """
    return gf.config.default_float()


# noinspection PyPep8Naming
class NP:
    """ Extended numpy types."""
    Array = np.ndarray
    Tensor = np.ndarray  # Generic Tensor.
    Tensor1 = Tensor    # Second Order Tensor, tf.shape = (i,j)
    Tensor2 = Tensor    # Second Order Tensor, tf.shape = (i,j)
    Vector = Tensor2    # First Order Tensor, column vector, tf.shape = (j,1)
    Covector = Tensor2    # First Order Tensor, row vector, tf.shape = (1,j)
    Matrix = Tensor2    # Second Order Tensor, tf.shape = (i,j)
    Tensor3 = Tensor    # Third Order Tensor, tf.shape = (i,j,k).
    Tensor4 = Tensor    # Fourth Order Tensor, tf.shape = (i,j,k,l).
    Tensor5 = Tensor
    Tensor6 = Tensor
    Tensor7 = Tensor
    Tensor8 = Tensor
    VectorLike = int | float | Sequence[int | float] | Array
    MatrixLike = VectorLike | Sequence[VectorLike]
    CovectorLike = MatrixLike
    ArrayLike = TensorLike = MatrixLike | Sequence[MatrixLike] | Sequence[Sequence[MatrixLike]]


# noinspection PyPep8Naming
class TF:
    """ Extended tensorflow types, and constants."""
    Array = tf.Tensor
    Tensor = tf.Tensor  # Generic Tensor.
    Tensor1 = Tensor    # Second Order Tensor, tf.shape = (i,j)
    Tensor2 = Tensor    # Second Order Tensor, tf.shape = (i,j)
    Vector = Tensor2    # First Order Tensor, column vector, tf.shape = (j,1)
    Covector = Tensor2    # First Order Tensor, row vector, tf.shape = (1,j)
    Matrix = Tensor2    # Second Order Tensor, tf.shape = (i,j)
    Tensor3 = Tensor    # Third Order Tensor, tf.shape = (i,j,k).
    Tensor4 = Tensor    # Fourth Order Tensor, tf.shape = (i,j,k,l).
    Tensor5 = Tensor
    Tensor6 = Tensor
    Tensor7 = Tensor
    Tensor8 = Tensor
    VectorLike = int | float | Sequence[int | float] | Array
    MatrixLike = Union[VectorLike, Sequence[VectorLike]]
    CovectorLike = MatrixLike
    ArrayLike = TensorLike = MatrixLike | Sequence[MatrixLike] | Sequence[Sequence[MatrixLike]]
    Slice = PairOfInts = tf.Tensor      #: A slice, for indexing and marginalization.

    NaN: TF.Tensor = tf.constant(np.NaN, dtype=FLOAT())     #: A constant Tensor representing NaN.

