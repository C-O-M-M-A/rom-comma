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

""" Contains Global Sensitivity Analysis tools."""

from __future__ import annotations

from abc import abstractmethod

from romcomma.typing_ import *
import gpflow as gf
import tensorflow as tf
from romcomma.base import Model, Parameters
from romcomma.gpr import GPInterface

class GSAInterface(Model, gf.Module):
    """ Interface encapsulating a Global Sensitivity Analysis. """

    class Parameters(Parameters):
        """ The Parameters set of a GSA."""

        @classmethod
        @property
        def Values(cls) -> Type[NamedTuple]:
            """ The NamedTuple underpinning this Parameters set."""

            class Values(NamedTuple):
                """ The parameters set of a GSA.

                Attributes:
                    theta (NP.Matrix): An (M,M) rotation matrix.
                    result (NP.Matrix): An (L, L, M) tensor result, flattened into an (L*L, M) matrix.
                    covariance (NP.Matrix): An (L, L, L, L, M, M) tensor covariance of result, flattened into an (L*L*L*L, M*M) matrix.
                """
                theta: NP.Matrix = np.atleast_2d(0.0)
                result: NP.Matrix = np.atleast_2d(0.0)
                covariance: NP.Matrix = np.atleast_2d(0.0)

            return Values

    @classmethod
    @property
    def DEFAULT_OPTIONS(cls) -> Dict[str, Any]:
        """ Calculation options"""
        return {'is_result_diagonal': True, 'is_covariance_partial': True, 'is_covariance_diagonal': True}

    @property
    def theta(self):
        return self._Theta

    @theta.setter
    def theta(self, value):
        self._Theta = value
        self.calculate()

    def __init__(self, name: str, gp: GPInterface, theta: TF.Matrix = None, **kwargs: Any):
        """ Construct a GSA object.

        Args:
            name: The name of the GSA.
            gp: The gp to analyze.
            theta: (M,M) input rotation matrix.
            **kwargs: Set calculation options, by updating DEFAULT_OPTIONS.
        """
        super(Model, self).__init__(name=name)
        self._gp = gp
        self._folder = self._gp.folder / 'gsa' / name
        self.options = self.DEFAULT_OPTIONS | kwargs
        if theta is None and self._folder.exists():
            super().__init__(folder=self._folder, read_parameters=True)
        else:
            theta = np.eye(self._gp.M, dtype = gf.config.default_float()) if theta is None else theta
            super().__init__(folder=self._folder, read_parameters=False, theta=theta)
        self._theta = tf.constant(self.params.theta, dtype=gf.config.default_float())

        # Unwrap parameters
        self._E = tf.constant(self._gp.likelihood.params.variance, dtype=gf.config.default_float())
        self._F = tf.constant(self._gp.kernel.params.variance, dtype=gf.config.default_float())
        self._Lambda = tf.constant(self._gp.kernel.params.lengthscales, dtype=gf.config.default_float())
        pass


class GSA(GSAInterface):
    """ Implementation of Global Sensitivity Analysis. """

