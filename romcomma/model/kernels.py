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

""" Contains Kernel base class and GPFlow implementations of kernels required for Gaussian Process Regression."""

from __future__ import annotations

from romcomma.typing_ import *
from abc import abstractmethod
from romcomma.model.base import Parameters, Model
from numpy import atleast_2d
import gpflow as gf


class Kernel(Model):
    """ Abstract interface to a Kernel. Essentially this is the code contract with the GP interface."""

    class Parameters(Parameters):
        """ Abstraction of the parameters of a Kernel."""

        @classmethod
        @property
        def Values(cls) -> Type[NamedTuple]:
            """ The NamedTuple underpinning this Parameters set."""

            class Values(NamedTuple):
                """ Abstraction of the parameters of a Kernel.

                Attributes:
                    variance: An (L,L) or (1,L) Matrix of kernel variances. (1,L) represents a diagonal (L,L) variance matrix.
                        (1,1) means a single kernel shared by all outputs.
                    lengthscales: A (V,M) Matrix of anisotropic lengthscales, or a (V,1) Vector of isotropic lengthscales,
                        where V=1 or V=variance.shape[1]*(variance.shape[0]+ 1)/2.
                """
                variance: NP.Matrix = atleast_2d(0.1)
                lengthscales: NP.Matrix = atleast_2d(0.2)

            return Values

    @classmethod
    @property
    def DEFAULT_OPTIONS(cls) -> Dict[str, Any]:
        """ **Do not use, this function is merely an interface requirement. **"""
        return {'A kernel has no use for optimizer options, only its parent GP does.': None}

    @classmethod
    @property
    def TYPE_IDENTIFIER(cls) -> str:
        """ The type of this Kernel object or class as '__module__.Kernel.__name__'."""
        return cls.__module__.split('.')[-1] + '.' + cls.__name__

    @classmethod
    def TypeFromIdentifier(cls, TypeIdentifier: str) -> Type[Kernel]:
        """ Convert a TypeIdentifier to a Kernel Type.

        Args:
            TypeIdentifier: A string generated by Kernel.TypeIdentifier().
        Returns:
            The type of Kernel that _TypeIdentifier specifies.
        """
        for KernelType in cls.__subclasses__():
            if KernelType.TYPE_IDENTIFIER == TypeIdentifier:
                return KernelType
        raise TypeError('Kernel.TypeIdentifier() of unrecognizable type.')

    @classmethod
    def TypeFromParameters(cls, parameters: Parameters) -> Type[Kernel]:
        """ Recognize the Type of a Kernel from its Parameters.

        Args:
            parameters: A Kernel.Parameters array to recognize.
        Returns:
            The type of Kernel that parameters defines.
        """
        for kernel_type in Kernel.__subclasses__():
            if isinstance(parameters, kernel_type.Parameters):
                return kernel_type
        raise TypeError('Kernel Parameters array of unrecognizable type.')

    @property
    def L(self) -> int:
        """ The output (Y) dimensionality, or 1 for a single kernel shared across all outputs."""
        return self._L

    @property
    def M(self) -> int:
        """ The input (X) dimensionality, or 1 for an isotropic kernel."""
        return self._M

    @property
    @abstractmethod
    def implementation(self) -> Tuple[Any, ...]:
        """ The implementation in_??? version of this Kernel, for use in the in_??? GP.
            If ``self.variance.shape == (1,L)`` an L-tuple of kernels is returned.
            If ``self.variance.shape == (L,L)`` a 1-tuple of multi-output kernels is returned.
        """

    def broadcast_parameters(self, variance_shape: Tuple[int, int], M, folder: Optional[PathLike] = None) -> Kernel:
        """ Broadcast this kernel to higher dimensions. Shrinkage raises errors, unchanged dimensions silently nop.
        A diagonal variance matrix broadcast to a square matrix is initially diagonal. All other expansions are straightforward broadcasts.
        Args:
            variance_shape: The new shape for the variance, must be (1, L) or (L, L).
            M: The number of input Lengthscales per output.
            folder: The file location, which is ``self.folder`` if ``folder is None`` (the default).
        Returns: ``self``, for chaining calls.
        Raises:
            IndexError: If an attempt is made to shrink a parameter.
        """
        if variance_shape != self.params.variance.shape:
            self.parameters.broadcast_value(model_name=str(self.folder), field="variance", target_shape=variance_shape, is_diagonal=True, folder=folder)
            self._L = variance_shape[1]
        if (self._L, M) != self.params.lengthscales.shape:
            self.parameters.broadcast_value(model_name=str(self.folder), field="lengthscales", target_shape=(self._L, M), is_diagonal=False, folder=folder)
            self._M = M
        return self

    def calculate(self):
        """ Calculate the kernel."""
        pass

    def optimize(self, method: str, options: Optional[Dict] = DEFAULT_OPTIONS):
        """ **Do not use, this function is merely an interface requirement. **

        Args:
            method: The optimization algorithm (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
            options: Dict of implementation-dependent optimizer options. options = None indicates that options should be read from JSON file.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError('A kernel cannot be implemented.')

    def __init__(self, folder: PathLike, read_parameters: bool = False, **kwargs: NP.Matrix):
        """ Construct a Kernel. This must be called as a matter of priority by all implementations.

        Args:
            folder: The model file location.
            read_parameters: If True, the model.parameters are read from ``folder``, otherwise defaults are used.
            **kwargs: The model.parameters fields=values to replace after reading from file/defaults.
        """
        super().__init__(folder, read_parameters, **kwargs)
        self._L, self._M = self.params.variance.shape[1], self.params.lengthscales.shape[1]
        self.broadcast_parameters(self.params.variance.shape, self._M)


class RBF(Kernel):
    """ Implements the RBF kernel_parameters for use with romcomma.model.implemented_in_gpflow."""

    @property
    def implementation(self) -> Tuple[Any, ...]:
        """ The implemented_in_??? version of this Kernel, for use in the implemented_in_??? GP.
            If ``self.variance.shape == (1,1)`` a 1-tuple of kernels is returned.
            If ``self.variance.shape == (1,L)`` an L-tuple of kernels is returned.
            If ``self.variance.shape == (L,L)`` a 1-tuple of multi-output kernels is returned.
        """
        if self.params.variance.shape[0] == 1:
            ard = (self.params.lengthscales.shape[1] == 1)
            results = tuple(gf.kernels.RBF(variance=self.params.variance[0, l], lengthscales=self.params.lengthscales[l])
                            for l in range(self.params.variance.shape[1]))
            # for result in results[:-1]:
            #     gpflow.set_trainable(result, False)
        else:
            raise NotImplementedError(f'Kernel.RBF is not implemented_in_gpflow for variance.shape={self.params.variance.shape}, ' +
                                      f'only for variance.shape=(1,{self.params.variance.shape[1]}) using independent gps.')
        return results
