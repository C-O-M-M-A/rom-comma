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

""" Wraps GPy proprietary package as a GaussianBundle.

Contents:
    :GaussianBundle: class.
"""

from romcomma.typing_ import Optional, NP, PathLike, NamedTuple, Tuple, Sequence
from romcomma.data import Fold, Store, Frame
from romcomma.model import base
from collections import namedtuple
from pathlib import Path
from numpy import atleast_1d, atleast_2d, transpose, array, einsum, full, broadcast_to
from pandas import concat
import GPy


class Kernel:

    # noinspection PyPep8Naming,PyPep8Naming
    class ExponentialQuadratic(base.Kernel):
    
        MEMORY_LAYOUT = 'C'
    
        Parameters = namedtuple("Parameters", ["f", "lengthscale", "ard", "lengthscale_trainer"])
        PARAMETER_DEFAULTS = Parameters(f=atleast_2d(1.0), lengthscale=atleast_2d(1.0), ard=atleast_2d(False), lengthscale_trainer=atleast_2d(0))
    
        def calculate(self):
            pass
    
        @property
        def gpy(self):
            return GPy.kern.RBF(self._M, self._parameters.f[0, 0], self._parameters.lengthscale[0, :], self._parameters.ard[0, 0])
    
        def __init__(self, X0: Optional[NP.Matrix], X1: Optional[NP.Matrix], dir_: PathLike = "", parameters: Optional[NamedTuple] = None,
                     overwrite: bool = False):
            super().__init__(X0, X1, dir_, parameters, overwrite)


# noinspection PyPep8Naming
class GaussianProcess(base.GaussianProcess):
    """ Implementation of a Gaussian Process."""

    MEMORY_LAYOUT = 'C'

    @property
    def f_derivative(self) -> NP.Matrix:
        return array(0)

    @property
    def e_derivative(self) -> NP.Matrix:
        return array(0)

    @property
    def inv_prior_Y_y(self) -> NP.Vector:
        return self._gpy.posterior.woodbury_vector

    @property
    def log_marginal_likelihood(self) -> float:
        return 0

    @property
    def posterior_Y(self) -> Tuple[NP.Vector, NP.Matrix]:
        return array(0), array(0)

    @property
    def posterior_F(self) -> Tuple[NP.Vector, NP.Matrix]:
        return array(0), array(0)

    def calculate(self):
        pass

    def parameter_update(self) -> NP.Covector:
        """Update GaussianProcess parameters but not _kernel
        Returns: The new _kernel lengthscale Covector.
        """
        self._parameters = self.Parameters(f=self._gpy.rbf.variance[0], e=self._gpy.Gaussian_noise.variance[0][0],
                                           kernel=self._kernel.type_identifier(), log_likelihood=-abs(self._gpy.log_likelihood()),
                                           lengthscale_trainer=self._parameters.lengthscale_trainer, e_floor=self._parameters.e_floor)
        return array(self._gpy.rbf.lengthscale)

    def predict(self, x: NP.Matrix, y_instead_of_f: bool = True) -> Tuple[NP.Vector, NP.Matrix]:
        result = self._gpy.predict(x, include_likelihood=y_instead_of_f)
        return result[0], broadcast_to(result[1], result[0].shape).copy()

    def predict_toy(self, x: NP.Matrix, y_instead_of_f: bool = True) -> NP.Vector:
        kern = self._gpy.kern.K(x, self.X)
        inv_prior_Y_y = self.inv_prior_Y_y
        result = self._gpy.predict(x, include_likelihood=y_instead_of_f)[0]
        result -= einsum('in, nj -> ij', kern, inv_prior_Y_y)
        return result

    def optimize(self, **kwargs):
        self._gpy.optimize(**kwargs)
        if self._gpy.Gaussian_noise.variance[0][0] < self.parameters.e_floor[0, 0]:
            self._gpy.rbf.variance = 1.0
            self._gpy.Gaussian_noise.variance = self.parameters.e_floor[0, 0]
            self._gpy.Gaussian_noise.variance.fix()
            self._gpy.optimize(**kwargs)
        self.parameter_update()

    def __init__(self, Y: NP.Matrix, kernel: base.Kernel, parameters: Optional[base.GaussianProcess.Parameters] = None):
        """ Construct a GaussianProcess.
        Args:
            kernel: A pre-constructed Kernel.
            Y: An ``NxL`` Response (label) matrix.
            parameters: A GaussianProcess.Parameters NamedTuple.
        """
        super().__init__(Y, kernel, parameters)
        self._gpy = GPy.models.GPRegression(self.X, self.Y, self._kernel.gpy, noise_var=self.parameters.e)
        self._gpy.kern.lengthscale.trainable = self.parameters.lengthscale_trainer[0, 0]


# noinspection PyPep8Naming
class GaussianBundle(base.GaussianBundle):
    """ Implementation of a Gaussian Process."""

    MEMORY_LAYOUT = 'C'

    # noinspection PyProtectedMember
    def parameter_update(self):
        if self._parameters.lengthscale_trainer.df.iloc[0, 0] < 0:
            self._parameters.f.df.iloc[0, 0] = self._gp[0].parameters.f
            self._parameters.e.df.iloc[0, 0] = self._gp[0].parameters.e
            self._parameters.log_likelihood.df.iloc[0, 0] = self._gp[0].parameters.log_likelihood
            trained_gp = self._gp[0]
        else:
            for l in range(self._L):
                self._parameters.f.df.iloc[0, l] = self._gp[l].parameters.f
                self._parameters.e.df.iloc[0, l] = self._gp[l].parameters.e
                self._parameters.log_likelihood.df.iloc[0, l] = self._gp[l].parameters.log_likelihood
            trained_gp = self._gp[self._parameters.lengthscale_trainer.df.iloc[0, 0]]
        self._parameters = self.write_frames(self._parameters)
        self._kernel.parameters = self._kernel.parameters._replace(f=atleast_2d(trained_gp._gpy.rbf.variance[0]),
                                                                   lengthscale=atleast_2d(trained_gp._gpy.rbf.lengthscale.values))
        self._kernel_template = type(self._kernel_template)(X0=None, X1=None, dir_=self._kernel_template.dir,
                                                            parameters=self._kernel.parameters, overwrite=True)

    def predict(self, x: NP.Matrix, y_instead_of_f: bool=True) -> NP.Tensor4:
        result = array([array(gp.predict(x, y_instead_of_f)) for gp in self._gp])
        if self._parameters.lengthscale_trainer.df.iloc[0, 0] < 0:
            result = transpose(result, [1, 2, 3, 0])
        else:
            result = transpose(result, [1, 2, 0, 3])
        result.shape = result.shape[:-1]
        return result

    def optimize(self, **kwargs):
        for gp in self._gp:
            if gp.parameters.lengthscale_trainer[0, 0]:
                gp.optimize(**kwargs)
                lengthscale = gp._gpy.rbf.lengthscale.values
        for gp in self._gp:
            if not gp.parameters.lengthscale_trainer[0, 0]:
                gp._gpy.rbf.lengthscale = lengthscale
                gp._gpy.rbf.lengthscale.fix()
                gp.optimize(**kwargs)
        self.parameter_update()

    def _validate_parameters(self):
        super()._validate_parameters()
        if self.parameters.f.df.shape[0] != 1:
            raise ValueError("GPy will not accept {0:d}x{1:d} f and e parameters. Use 1x1 or 1x{0:d} (diagonal)."
                             .format(self.parameters.f.df.shape[0], self.parameters.f.df.shape[1]))

    def __init__(self, fold: Fold, name: str, parameters: Optional[base.GaussianBundle.Parameters] = None, overwrite: bool = False,
                 reset_log_likelihood: bool = True):
        """ GaussianBundle constructor
        Args:
            fold: The location of the Model.GaussianProcess. Must be a fold.
            name: The name of this Model.GaussianProcess
            parameters: The model parameters. If ``None`` these are read from ``fold/name``.
                Otherwise these are written to ``fold/name``.
                Each parameter is a covariance matrix, provided as a square Matrix, or a CoVector if diagonal.
            overwrite: If True, any existing directory named ``fold/name`` is deleted.
                Otherwise no existing files are overwritten.
        """
        super().__init__(fold, name, parameters, overwrite, reset_log_likelihood)
        self._kernel = type(self._kernel_template)(X0=self.X, X1=self.X, dir_="", parameters=self._kernel_template.parameters_as_matrices)
        if self._parameters.lengthscale_trainer.df.iloc[0, 0] < 0:
            params = self._parameters._replace(lengthscale_trainer=atleast_2d(True),
                                               e_floor=atleast_2d(self._parameters.e_floor.df.iloc[0, 0]),
                                               f=atleast_2d(self._parameters.f.df.iloc[0, 0]), e=atleast_2d(self._parameters.e.df.iloc[0, 0]))
            self._gp.append(GaussianProcess(self.Y, self._kernel, params))
        else:
            for l in range(self._L):
                params = self._parameters._replace(lengthscale_trainer=atleast_2d(self._parameters.lengthscale_trainer.df.iloc[0, 0] == l),
                                                   e_floor=atleast_2d(self._parameters.e_floor.df.iloc[0, 0]),
                                                   f=atleast_2d(self._parameters.f.df.iloc[0, l]), e=atleast_2d(self._parameters.e.df.iloc[0, l]))
                self._gp.append(GaussianProcess(self.Y[:, [l]], self._kernel, params))


# noinspection PyPep8Naming
class Sobol(base.Sobol):

    MEMORY_LAYOUT = 'C'

