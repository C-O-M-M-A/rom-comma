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

""" Contains Kernel classes for gpr. """

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.base.classes import Data, Model


class Kernel(Model):
    """ Abstract interface to a Kernel. Essentially this is the code contract with the MOGP interface."""

    class Data(Data):
        """ The Data set of a Kernel."""

        @classmethod
        @property
        def NamedTuple(cls) -> Type[NamedTuple]:

            class Values(NamedTuple):
                """ The data set of a Kernel.

                Attributes:
                    variance: An (L,L) or (1,L) Matrix of kernel variances. (1,L) represents a diagonal (L,L) variance matrix.
                        (1,1) means a single kernel shared by all outputs.
                    lengthscales: A (L,M) Matrix of anisotropic lengthscales, or a (L,1) Vector of isotropic lengthscales,
                        where L=1 or L=variance.shape[1]*(variance.shape[0]+ 1)/2.
                """
                variance: NP.Matrix = np.atleast_2d(0.5)
                lengthscales: NP.Matrix = np.atleast_2d(0.5)

            return Values

    @classmethod
    @property
    def META(cls) -> Dict[str, Any]:
        return {'variance': True, 'covariance': False, 'lengthscales': {'variant': True, 'covariant': False}}

    def optimize(self, **kwargs: Any) -> Dict[str, Any]:
        """ Merely sets which data are trainable. """
        options = self.META | kwargs
        if self.is_covariant:
            gf.set_trainable(self._implementation[0].variance._cholesky_diagonal, options['variance'])
            gf.set_trainable(self._implementation[0].variance._cholesky_lower_triangle, options['covariance'])
            gf.set_trainable(self._implementation[0].lengthscales, options['lengthscales']['covariant'])
        else:
            for implementation in self._implementation:
                gf.set_trainable(implementation.variance, options['variance'])
                gf.set_trainable(implementation.lengthscales, options['lengthscales']['variant'])
        return options

    @classmethod
    @property
    def TYPE_IDENTIFIER(cls) -> str:
        """ The type of this Kernel object or class as '__module__.Kernel.__name__'."""
        return cls.__module__.split('.')[-1] + '.' + cls.__name__

    @classmethod
    def TypeFromIdentifier(cls, TypeIdentifier: str) -> Type[Kernel]:
        """ Convert a TypeIdentifier to a Kernel NamedTuple.

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
    def TypeFromParameters(cls, parameters: Data) -> Type[Kernel]:
        """ Recognize the NamedTuple of a Kernel from its Data.

        Args:
            parameters: A Kernel.Data array to recognize.
        Returns:
            The type of Kernel that data defines.
        """
        for kernel_type in cls.__subclasses__():
            if isinstance(parameters, kernel_type.Data):
                return kernel_type
        raise TypeError('Kernel Data array of unrecognizable type.')

    @property
    def L(self) -> int:
        """ The output (Y) dimensionality, or 1 for a single kernel shared across all outputs."""
        return self._L

    @property
    def M(self) -> int:
        """ The input (X) dimensionality, or 1 for an isotropic kernel."""
        return self._M

    @property
    def is_covariant(self) -> bool:
        """ Whether the kernel is covariant between outputs. """
        return self.params.variance.shape[0] > 1

    def broadcast_parameters(self, variance_shape: Tuple[int, int], M, folder: Optional[Path | str] = None) -> Kernel:
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
        self._implementation = None
        self._implementation = self.implementation
        return self

    @property
    @abstractmethod
    def implementation(self) -> Tuple[Any, ...]:
        """ The implementation of this Kernel, for use in MOGP.implementation.
            If ``self.variance.shape == (1,L)`` an L-tuple of kernels is returned.
            If ``self.variance.shape == (L,L)`` a 1-tuple of multi-output kernels is returned.
        """

    def __init__(self, folder: Path | str, read_data: bool = False, **kwargs: NP.Matrix):
        """ Construct a Kernel. This must be called as a matter of priority by all implementations.

        Args:
            folder: The model file location.
            read_data: If True, the data are read from ``folder``, otherwise defaults are used.
            **kwargs: The model.data fields=values to replace after reading from file/defaults.
        """
        super().__init__(folder, read_data, **kwargs)
        self._L, self._M = self.params.variance.shape[1], self.params.lengthscales.shape[1]
        self.broadcast_parameters(self.params.variance.shape, self._M)


class RBF(Kernel):

    @property
    def implementation(self) -> Tuple[Any, ...]:
        """ The implemented_in_??? version of this Kernel, for use in the implemented_in_??? MOGP.
            If ``self.variance.shape == (1,1)`` a 1-tuple of kernels is returned.
            If ``self.variance.shape == (1,L)`` an L-tuple of kernels is returned.
            If ``self.variance.shape == (L,L)`` a 1-tuple of multi-output kernels is returned.
        """
        if self._implementation is None:
            if self.params.variance.shape[0] == 1:
                self._implementation = tuple(gf.kernels.RBF(variance=max(self.params.variance[0, l], 1.0005E-6), lengthscales=self.params.lengthscales[l])
                                for l in range(self.params.variance.shape[1]))
            else:
                self._implementation = (mf.kernels.RBF(variance=self.params.variance, lengthscales=self.params.lengthscales), )
        return self._implementation
