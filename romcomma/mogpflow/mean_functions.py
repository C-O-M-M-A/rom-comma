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

""" Mean functions for mogpflow - i.e. Gaussian prior predictions."""

from __future__ import annotations

from typing import Sequence, Optional, Union
from gpflow.config import default_float
from gpflow.mean_functions import MeanFunction, Zero
import tensorflow as tf

class MOMeanFunction(MeanFunction):
    """ Mean functions for MOGPR. Basically a wrapper for a Sequence of gpflow.mean_functions.MeanFunctions, one for each output_dim.
    These functions constitute the prior mean predictions f(x) in the absence of any training data.
    """

    @property
    def output_dim(self):
        """ Also known as L."""
        return len(self._functions)

    @property
    def L(self):
        return self.output_dim

    @property
    def functions(self):
        """ The sequence of functions defining this MOMeanFunction."""
        return self._functions

    def __call__(self, X):
        """ Given N datapoints in X, returns an output_dim * N vector of flatten(functions(X))."""
        return tf.concat([f(X) for f in self._functions], axis=0)

    def __init__(self, output_dim: int, mean_functions: Union[MOMeanFunction, MeanFunction, Sequence[MeanFunction]] = Zero):
        """

        Args:
            output_dim: The number of mean_functions required, also known as L.
            mean_functions: Is broadcast to an L-Sequence of functions, giving the prior mean f(x) for each output_dim in turn.
        """
        if isinstance(mean_functions, MOMeanFunction):
            mean_functions = mean_functions.functions
        elif isinstance(mean_functions, MeanFunction):
            mean_functions = (mean_functions,) * output_dim
        self._functions = mean_functions
