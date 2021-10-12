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

""" Adds extra Types to the standard library typing package."""

import numpy
# noinspection PyPep8Naming
import tensorflow as TF
from typing import *
from pathlib import Path

EFFECTIVELY_ZERO = 1.0E-64

Numeric = Union[int, float]
OneOrMoreInts = Union[int, Sequence[int]]
ZeroOrMoreInts = Optional[OneOrMoreInts]
OneOrMoreFloats = Union[float, Sequence[float]]
ZeroOrMoreFloats = Optional[OneOrMoreFloats]
Module = type(numpy)

PathLike = Union[str, Path]
Unused = TF.exp


# noinspection PyPep8Naming
class NP:
    """ Extended numpy types."""
    Array = numpy.ndarray
    Tensor = numpy.ndarray  # Generic Tensor.
    Tensor4 = Tensor    # Fourth Order Tensor, ndarray.shape = (i,j,k,l).
    Tensor3 = Tensor    # Third Order Tensor, ndarray.shape = (i,j,k).
    Matrix = Tensor    # Second Order Tensor, ndarray.shape = (i,j)
    Vector = Tensor    # First Order Tensor, column vector, ndarray.shape = (j,1)
    Covector = Tensor    # First Order Tensor, row vector, ndarray.shape = (1,j)
    VectorLike = Union[Numeric, Sequence[Numeric], Array]
    MatrixLike = Union[VectorLike, Sequence[VectorLike]]
    CovectorLike = MatrixLike
    ArrayLike = TensorLike = Union[MatrixLike, Sequence[MatrixLike], Sequence[Sequence[MatrixLike]]]
    VectorOrMatrix = TypeVar('VectorOrMatrix', Vector, Matrix)
