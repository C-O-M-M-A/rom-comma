""" Adds extra Types to the standard library typing package."""

import numpy
from typing import *
from pathlib import Path

Numeric = Union[int, float]
OneOrMoreInts = Union[int, Sequence[int]]
ZeroOrMoreInts = Optional[OneOrMoreInts]
OneOrMoreFloats = Union[float, Sequence[float]]
ZeroOrMoreFloats = Optional[OneOrMoreFloats]
Cls = Any
Module = type(numpy)

PathLike = Union[str, Path]


# noinspection PyPep8Naming
class NP:
    """ Extended numpy types. """

    VectorLike = Union[Numeric, Sequence[Numeric]]
    MatrixLike = Union[VectorLike, Sequence[VectorLike]]
    ArrayLike = TensorLike = Union[MatrixLike, Sequence[MatrixLike], Sequence[Sequence[MatrixLike]]]
    Array = numpy.ndarray
    Tensor = numpy.ndarray  # Generic Tensor.
    Tensor4 = Tensor    # Fourth Order Tensor, ndarray.shape = (i,k,k).
    Tensor3 = Tensor    # Third Order Tensor, ndarray.shape = (i,j,k).
    Matrix = Tensor    # Second Order Tensor, ndarray.shape = (i,j)
    Vector = Tensor    # First Order Tensor, column vector, ndarray.shape = (j,1)
    Covector = Tensor    # First Order Tensor, row vector, ndarray.shape = (1,j)
