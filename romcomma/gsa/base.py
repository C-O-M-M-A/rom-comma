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
#  PROCUREMENT OF SUBSTITUTE G00DS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Base Classes and basic functions underpinning GSA."""

from __future__ import annotations

from romcomma.base.definitions import *
from abc import ABC


def det(variance_cho_diagonal):
    return tf.reduce_prod(variance_cho_diagonal, axis=-1)


def pdf(exponent: TF.Tensor, variance_cho_diagonal: TF.Tensor):
    """ Calculate the Gaussian pdf from the output of Gaussian.log_pdf.
    Args:
        exponent: The exponent in the Gaussian pdf.
        variance_cho_diagonal: The diagonal of the variance Cholesky decomposition.

    Returns: The Gaussian pdf.
    """
    return tf.exp(exponent) / det(variance_cho_diagonal)


def log_pdf(mean: TF.Tensor, variance_cho: TF.Tensor, is_variance_diagonal: bool,
            ordinate: TF.Tensor = tf.constant(0, dtype=FLOAT()), LBunch: int = 2) -> Any:
    """ Computes the logarithm of the un-normalized gaussian probability density, and the broadcast diagonal of variance_cho.
    Taking the product ``2 * Pi * Gaussian.det(variance_cho_diagonal)`` gives the normalization factor for the gaussian pdf.
    Batch dimensions of ordinate, mean and variance are internally broadcast to match each other.
    This function is used to minimize exponentiation, for efficiency and accuracy purposes, in calculating ratios of gaussian pdfs.

    Args:
        mean: Gaussian population mean. Should be of adequate rank to broadcast Ls.
        variance_cho: The lower triangular Cholesky decomposition of the Gaussian population variance. Is automatically broadcast to embrace Ns
        is_variance_diagonal: True if variance is an M-vector
        ordinate: The ordinate (z-value) to calculate the Gaussian density for. Should be of adequate rank to broadcast Ls. If not supplied, 0 is assumed.
        LBunch: The number of consecutive output (L) dimensions to count before inserting an N for broadcasting. Usually 2, sometimes 3.
    Returns: The tensor Gaussian pdf, and the diagonal of variance_cho.
    """
    # Broadcast ordinate - mean.
    if ordinate.shape == mean.shape:
        shape = ordinate.shape.as_list()
        fill = [1, ] * (len(shape) - 1)
        ordinate = tf.reshape(ordinate, shape[:-1] + fill + [shape[-1]])
        mean = tf.reshape(mean, fill + shape)
    ordinate = ordinate - mean
    # Broadcast variance_cho
    insertions = (tf.rank(variance_cho) - (1 if is_variance_diagonal else 2))
    insertions -= insertions % LBunch
    for axis in range(insertions, 0, -LBunch):
        variance_cho = tf.expand_dims(variance_cho, axis=axis)
    # Calculate the Gaussian pdf.
    if is_variance_diagonal:
        exponent = ordinate / tf.broadcast_to(variance_cho, tf.concat([variance_cho.shape[:-2], ordinate.shape[-2:]], axis=0))
    else:
        exponent = tf.squeeze(tf.linalg.triangular_solve(variance_cho, ordinate[..., tf.newaxis], lower=True), axis=-1)
    exponent = - 0.5 * tf.einsum('...o, ...o -> ...', exponent, exponent)
    return exponent, variance_cho if is_variance_diagonal else tf.linalg.diag_part(variance_cho)


class Calibrator(ABC):
    """ Interface to GSA calibrator"""

    @abstractmethod
    def marginalize(self, m: TF.Slice) -> Dict[str, TF.Tensor]:
        raise NotImplementedError('This is an abstract class.')


class Gaussian:
    """ Encapsulates a Gaussian pdf. For numerical stability the 2 Pi factor is not included."""

    exponent: TF.Tensor  #: The exponent of a Gaussian pdf, :math:`- z^{\intercal} \Sigma^{-1} z / 2`.
    cho_diag: TF.Tensor    #: The diagonal of the Cholesky decomposition of the Gaussian covariance.

    @property
    def det(self) -> TF.Tensor:
        """ The sqrt of the determinant of the Gaussian covariance. """
        return tf.reduce_prod(self.cho_diag, axis=-1)

    @property
    def pdf(self) -> TF.Tensor:
        """ Calculate the Gaussian pdf from the output of Gaussian.log_pdf."""
        return tf.exp(self.exponent) / self.det

    def expand_dims(self, axes: Sequence[int]) -> Gaussian:
        """ Insert dimensions at the specified axes.

        Args:
            axes: A sequence of dims to insert.
        Returns: ``self`` for chaining calls.
        """
        for axis in sorted(axes, reverse=True):
            self.exponent = tf.expand_dims(self.exponent, axis)
            self.cho_diag = tf.expand_dims(self.cho_diag, (axis - 1) if axis < 0 else axis)
        return self

    def __itruediv__(self, other) -> Gaussian:
        """ Divide this Gaussian pdf by denominator.

        Args:
            other: The Gaussian to divide by.
        """
        self.exponent -= other.exponent
        self.cho_diag /= other.cho_diag
        return self

    def __init__(self, mean: TF.Tensor, variance: TF.Tensor, is_variance_diagonal: bool, ordinate: TF.Tensor = tf.constant(0, dtype=FLOAT()), LBunch: int = 2):
        """ Computes the logarithm of the un-normalized gaussian probability density, and the broadcast diagonal of variance_cho.
        Taking the product ``2 * Pi * Gaussian.det(variance_cho_diagonal)`` gives the normalization factor for the gaussian pdf.
        Batch dimensions of ordinate, mean and variance are internally broadcast to match each other.
        This function is used to minimize exponentiation, for efficiency and accuracy purposes, in calculating ratios of gaussian pdfs.

        Args:
            mean: Gaussian population mean. Should be of adequate rank to broadcast Ls.
            variance: The Gaussian population variance. Is automatically broadcast to embrace Ns.
            is_variance_diagonal: True if variance is an M-vector.
            ordinate: The ordinate (z-value) to calculate the Gaussian density for. Should be of adequate rank to broadcast Ls. If not supplied, 0 is assumed.
            LBunch: The number of consecutive output (L) dimensions to count before inserting an N for broadcasting. Usually 2, sometimes 3.
        Returns: The tensor Gaussian pdf, and the diagonal of variance_cho.
        """
        # Broadcast ordinate - mean.
        variance_cho = tf.sqrt(variance) if is_variance_diagonal else tf.linalg.cholesky(variance)
        if ordinate.shape == mean.shape:
            shape = ordinate.shape.as_list()
            fill = [1, ] * (len(shape) - 1)
            ordinate = tf.reshape(ordinate, shape[:-1] + fill + [shape[-1]])
            mean = tf.reshape(mean, fill + shape)
        ordinate = ordinate - mean
        # Broadcast variance_cho
        insertions = (tf.rank(variance_cho) - (1 if is_variance_diagonal else 2))
        insertions -= insertions % LBunch
        for axis in range(insertions, 0, -LBunch):
            variance_cho = tf.expand_dims(variance_cho, axis=axis)
        # Calculate the Gaussian pdf.
        if is_variance_diagonal:
            exponent = ordinate / tf.broadcast_to(variance_cho, tf.concat([variance_cho.shape[:-2], ordinate.shape[-2:]], axis=0))
        else:
            exponent = tf.squeeze(tf.linalg.triangular_solve(variance_cho, ordinate[..., tf.newaxis], lower=True), axis=-1)
        exponent = - 0.5 * tf.einsum('...o, ...o -> ...', exponent, exponent)
        self.exponent = exponent
        self.cho_diag = variance_cho if is_variance_diagonal else tf.linalg.diag_part(variance_cho)


def sym_check(tensor: TF.Tensor, transposition: List[int]) -> TF.Tensor:
    return tf.reduce_sum((tensor - tf.transpose(tensor, transposition))**2)


def mean(tensor: TF.Tensor):
    n = tf.cast(tf.reduce_prod(tensor.shape), FLOAT())
    return tf.divide(tf.reduce_sum(tensor), n)


def sos(tensor: TF.Tensor, ein: str = 'lijk, lijk'):
    return tf.einsum(ein, tensor, tensor)


def ms(tensor: TF.Tensor, ein: str = 'lijk'):
    n = tf.cast(tf.reduce_prod(tensor.shape), FLOAT())
    return tf.divide(sos(tensor, ein), n)


def rms(tensor: TF.Tensor, ein: str = 'lijk, lijk'):
    return tf.sqrt(ms(tensor, ein))


def det(tensor: TF.Tensor):
    return tf.reduce_prod(tensor, axis=-1)

I = [0, 0, 0, 0]
