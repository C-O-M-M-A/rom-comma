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

""" Test functions, taken from `SALib <https://salib.readthedocs.io/en/latest/api/SALib.test_functions.html>_"""

from __future__ import annotations

from romcomma.base.definitions import *
import SALib.test_functions.Ishigami, SALib.test_functions.Sobol_G, SALib.test_functions.oakley2004


class Scalar:
    """A scalar function ``scalar`` such that ``scalar(X, kwargs)`` calls ``self.call(self.loc + self.scale * X[:, :self.m], **(self.kwargs | kwargs)``."""

    @property
    def call(self) -> Callable[..., float]:
        return self._call

    @property
    def loc(self) -> NP.VectorLike:
        return self._loc

    @property
    def scale(self) -> NP.VectorLike:
        return self._scale

    @property
    def m(self) -> int:
        return self._m

    @property
    def kwargs(self) -> NP.VectorLike:
        return self._kwargs

    def __call__(self, X: NP.Matrix, **kwargs) -> NP.Matrix:
        return np.reshape(self._call(self._loc + self._scale * X[:, :self._m], **(self._kwargs | kwargs)), (X.shape[0], 1))

    def __init__(self, call: Callable[NP.Matrix, float], loc: NP.VectorLike, scale: NP.VectorLike, m: int, **kwargs):
        """ A scalar function, which calls ``call(loc + scale * X[:, :m], **kwargs)``.

        Args:
            call: This function called.
            loc: Input offset.
            scale: Input scale.
            m: The number of input dimensions.
            **kwargs: Function parameters applied to call.
        """
        self._call = call
        self._loc = loc
        self._scale = scale
        self._m = m
        self._kwargs = kwargs


class Vector(dict):
    """ A vector functon, which is little more than a named dictionary of Scalar functions, such that ``vector(X, **kwargs)`` concatenates
    ``scalar(X, **kwargs)`` for each dictionary item ``key: Scalar``. """

    @classmethod
    def concat(cls, name: str, vectors: Sequence[Vector]) -> Vector:
        """ Concatenate vectors.

        Args:
            name: The name of the returned ``Vector``.
            vectors: A sequence of ``Vector`` functions to concatenate.

        Returns: The concatenation of ``vectors``, named ``name``.

        """
        result = cls(name)
        for vector in vectors:
            result.update({f'{vector.name}.{key}': scalar for key, scalar in vector.items()})
        return result

    @property
    def name(self) -> str:
        return self._name

    @property
    def meta(self) -> Dict:
        """ Meta data for providing to ``data.storage``."""
        return {'name': self.name, 'call': {l: function for l, function in enumerate(self.keys())}}

    def subVector(self, name:str, scalars: Sequence[str]) -> Vector:
        """ Create a subVector of ``self``.

        Args:
            name: The name of the ``subVector``.
            scalars: The keys of the items of ``self`` to be included in subVector.
        Returns: A new instance of ``Vector`` named ``name`` containing the ``Scalars`` keyed ``scalars``. Effectively the pseudo-slice ``self[scalars]``.
        """
        return Vector(name, **{scalar: self[scalar] for scalar in scalars})

    def __call__(self, X: NP.Matrix, **kwargs) -> NP.Matrix:
        return np.concatenate([scalar(X, **kwargs) for scalar in self.values()], axis=1)

    def __init__(self, name: str, **kwargs: Scalar):
        """ Construct a vector function.

        Args:
            name: The name of this ``Vector``.
            **kwargs: The Dict of ``Scalar``s comprising this ``Vector``.
        """
        super().__init__(**kwargs)
        self._name = name


_ISHIGAMI = {'call': SALib.test_functions.Ishigami.evaluate, 'loc': -np.pi, 'scale': 2 * np.pi}     #: The Ishigami function without parameters.
_SOBOL_G = {'call': SALib.test_functions.Sobol_G.evaluate, 'loc': 0, 'scale': 1}    #: Modified Sobol G-function without parameters.
_OAKLEY2004 = {'call': SALib.test_functions.oakley2004.evaluate, 'loc': -1, 'scale': 2}     #: Modified Oakley & O'Hagan (2004) function without parameters.


def linspace(start: float, stop: float, shape: Sequence[int]) -> NP.Matrix:
    """ A multi-dimensional version of ``np.linspace``, distributing values throughout ``shape``.

    Args:
        start: Start value, which will be returned in ``linspace(...)[0,...,0]``.
        stop: Stop value, which will be returned in ``linspace(...)[-1,...,-1]``.
        shape: The ``linspace.shape`` to return.
    Returns: ``np.reshape(np.linspace(start, stop, int(np.prod(shape)), endpoint=True), newshape=shape)``.
    """
    return np.reshape(np.linspace(start, stop, int(np.prod(shape)), endpoint=True), newshape=shape)


ISHIGAMI = Vector(name='ishigami',
                  standard=Scalar(**_ISHIGAMI, m=3, A=7.0, B=0.1),
                  balanced=Scalar(**_ISHIGAMI, m=3, A=20.0, B=1.0),
                  sin=Scalar(**_ISHIGAMI, m=3, A=0.0, B=0.0),
                  ) #: 3 example Ishigami functions, requiring ``M >= 3``.


SOBOL_G = Vector(name='sobol_g',
                 weak5_2=Scalar(**_SOBOL_G, m=5, a=np.array([3, 6, 9, 18, 27]), alpha=np.ones((5,)) * 2.0),
                 strong5_2=Scalar(**_SOBOL_G, m=5, a=np.array([1/2, 1, 2, 4, 8]), alpha=np.ones((5,)) * 2.0),
                 strong5_4=Scalar(**_SOBOL_G, m=5, a=np.array([1/2, 1, 2, 4, 8]), alpha=np.ones((5,)) * 4.0),
                 ) #: 3 example modified Sobol G-functions, requiring ``M >= 3``.


OAKLEY2004 = Vector(name='oakley2004',
                    lin7=Scalar(**_OAKLEY2004, m=7, A=[linspace(start=7.0, stop=7.0 / 2, shape=[7, ]), ] + [np.zeros([7])] * 2,
                                M=np.zeros([7, 7])),
                    quad7=Scalar(**_OAKLEY2004, m=7, A=[linspace(start=7.0, stop=7.0 / 2, shape=[7, ]), ] + [np.zeros([7])] * 2,
                                 M=linspace(start=7.0, stop=1.0, shape=[7, 7])),
                    balanced_quad7=Scalar(**_OAKLEY2004, m=7, A=[-linspace(start=7.0, stop=7.0 / 2, shape=[7, ]), ] + [np.zeros([7])] * 2,
                                          M=linspace(start=1.0, stop=7.0, shape=[7, 7])),
                    ) #: 3 example modified Oakley & O'Hagan (2004) functions, requiring ``M >= 7``.


ALL = Vector.concat(name='all', vectors=(ISHIGAMI, SOBOL_G, OAKLEY2004))    #: The concatenation of ISHIGAMI, SOBOL_G, OAKLEY2004.

