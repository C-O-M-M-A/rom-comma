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

""" **User interface for undertaking GSA on a MOGP** """

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.base.classes import Model, Data, Frame
from romcomma.gpr.models import GPR
from romcomma.gsa.base import Calibrator
from romcomma.gsa.calculators import ClosedSobol, ClosedSobolWithError
from enum import IntEnum, auto
from abc import abstractmethod


class GSA(Model):
    """ Class encapsulating a generic Sobol calculation."""

    class Kind(IntEnum):
        """ Enum to specify the kind of Sobol index to calculate."""
        FIRST_ORDER = auto()
        CLOSED = auto()
        TOTAL = auto()

    @classmethod
    @property
    def ALL_KINDS(cls) -> List[GSA.Kind]:
        return [kind for kind in cls.Kind]

    @staticmethod
    def _columns(M: int, m_cols: int, m_list: List[int]) -> pd.Index:
        """ ClosedSobol pd.DataFrame columns for output.

        Args:
            M: The dimensionality of the full model.
            m_cols: The number of input columns covering ``m``.
            m_list: The list of input columns covering ``m``.
        Returns: A pd.ClosedSobol of column labels (integers).
        """
        if m_cols > len(m_list):
            m_list = m_list + [M]
        if m_cols > len(m_list):
            m_list = [-1] + m_list
        return pd.Index(m_list, name='m')

    @staticmethod
    def _index(shape: List[int]) -> pd.MultiIndex:
        """ ClosedSobol a pd.DataFrame for output.

        Args:
            shape: The shape of the pd.DataFrame to index.
        Returns: The pd.MultiIndex to index the pd.DataFrame
        """
        shape = shape[:-1]
        indices = [list(range(l)) for l in shape]
        return pd.MultiIndex.from_product(indices, names=[f'l.{l}' for l in range(len(indices))])

    @property
    def _m_dataset(self) -> tf.data.Dataset:
        """ Returns a dataset of slices to iterate through to perform th GSA.
        """
        m, M = self.meta['m'], self.meta['M']
        result = []
        ms = range(M) if m < 0 else [m]
        if self.kind == GSA.Kind.FIRST_ORDER:
            result = [tf.constant([m, m + 1], dtype=INT()) for m in ms]
        elif self.kind == GSA.Kind.CLOSED:
            result = [tf.constant([0, m + 1], dtype=INT()) for m in ms]
        elif self.kind == GSA.Kind.TOTAL:
            result = [tf.constant([m + 1, M], dtype=INT()) for m in ms]
        return tf.data.Dataset.from_tensor_slices(result)

    @property
    @abstractmethod
    def calibrator(self) -> Calibrator:
        """The object to do the calculations, whose marginalise(m) method returns a dict of results. """
        raise NotImplementedError('This is a base class. Derived classes must implement calibrator(cls, gp: GPR, is_error_calculated: bool, **kwargs: Any).')

    @abstractmethod
    def _post_calibrate(self, calibrator: Calibrator, results: Dict[str, TF.Tensor]) -> Dict[str, TF.Tensor]:
        raise NotImplementedError('This is a base class.')

    def _compose_and_save(self, results: Dict[str, TF.Tensor]):
        """ Compose and Save a GSA results tf.Tensor.

        Args:
            results: The path to save to.
        """
        m, M = self.meta['m'], self.meta['M']
        m_list = list(range(M)) if m < 0 else [m]
        for key, value in self.data.asdict().items():
            result = results.get(key, None)
            if result is not None:
                shape = result.shape.as_list()
                result = pd.DataFrame(tf.reshape(result, [-1, shape[-1]]).numpy(), columns=GSA._columns(M, shape[-1], m_list), index=GSA._index(shape))
                Frame(value.csv, result, float_format='%.6f')

    def calibrate(self, method: str = None, **kwargs) -> Dict[str, Any]:
        """ Perform a generic GSA calculation. This method should be overriden by specific subclasses, and called via ``super()`` as a matter of priority.

        Args:
            method: Not used.
        Returns: The results of the calculation, as a labelled dictionary of tf.Tensors.
        """
        m_dataset = self._m_dataset
        calibrator = self.calibrator
        first_iteration = True
        for m in m_dataset:
            result = calibrator.marginalize(m)
            if first_iteration:
                results = {key: value[..., tf.newaxis] for key, value in result.items()}
                first_iteration = False
            else:
                for key in results.keys():
                    results[key] = tf.concat([results[key], result[key][..., tf.newaxis]], axis=-1)
        results = self._post_calibrate(calibrator, results)
        self._compose_and_save(results)
        return self.meta

    def __init__(self, gp: GPR, kind: GSA.Kind, m: int = -1, is_error_calculated: bool = False, **kwargs: Any):
        """ Perform a general GSA. The object created is single use and disposable: the constructor performs and records the entire GSA and the
        constructed object is basically useless once constructed.

        Args:
            gp: The underlying Gaussian Process.
            kind: The kind of index to calculate - first order, closed or total.
            m: The final index of the reduced model. For a single calculation it is required that ``0 &le m &lt gp.M``.
                Any m outside this range results the Sobol index of kind being calculated for all ``m in range(1, M+1)``.
            is_error_calculated: Whether to calculate the standard error on the Sobol index.
                This is a memory intensive calculation, so leave this flag False unless you are sure you need errors
            **kwargs: The calculation meta to override META.
        """
        self.gp = gp
        self.is_error_calculated = is_error_calculated
        self.kind = kind
        m = m if 0 <= m < gp.M else -1
        name = kind.name.lower() if m == -1 else f'{kind.name.lower()}.{m}'
        folder = gp.folder / 'gsa' / name
        super().__init__(folder, read_data=False)
        self.meta = {'folder': str(folder), 'm': m, 'M': gp.M} | self.META | kwargs
        self.write_meta(self.meta)


class Sobol(GSA):
    """ Class encapsulating a generic Sobol calculation."""

    class Data(Data):
        """ The Data set of a GSA."""

        @classmethod
        @property
        def NamedTuple(cls) -> Type[NamedTuple]:
            """ The NamedTuple underpinning this Data set."""

            class Values(NamedTuple):
                """ The data set of a GSA.

                    Attributes:
                        S (NP.Matrix): The Sobol index.
                        T (NP.Matrix): The Sobol index StdDev.
                        V (NP.Matrix): The conditional variances underpinning the Sobol index.
                        W (NP.Matrix): The covariances underpinning the Sobol index variance.
                """
                S: NP.Matrix = np.atleast_2d(None)
                T: NP.Matrix = np.atleast_2d(None)
                V: NP.Matrix = np.atleast_2d(None)
                W: NP.Matrix = np.atleast_2d(None)

            return Values

    @classmethod
    @property
    def META(cls) -> Dict[str, Any]:
        """ Default calculation meta. ``is_T_partial`` forces ``WmM = 0``."""
        return ClosedSobolWithError.META

    @property
    def calibrator(self) -> ClosedSobol:
        """The object to do the calculations, whose marginalise(m) method returns a dict of results.
        Args:
            gp: The GPR underpinning the GSA.
            is_error_calculated: Whether to calculate the standard error of the GSA
            **kwargs: Options passed straight to the Calibrator.
        Returns: The Calibrator.
        """
        return ClosedSobolWithError(self.gp, **self.meta) if self.is_error_calculated else ClosedSobol(self.gp, **self.meta)

    def _post_calibrate(self, calibrator: Calibrator, results: Dict[str, TF.Tensor]) -> Dict[str, TF.Tensor]:
        results['V'] = tf.concat([results['V'], calibrator.V[0][..., tf.newaxis]], axis=-1)
        results['S'] = calibrator.S[..., tf.newaxis] - results['S'] if self.kind == GSA.Kind.TOTAL else results['S']
        results['S'] = tf.concat([results['S'], calibrator.S[..., tf.newaxis]], axis=-1)
        if 'T' in results and not self.meta['is_T_partial']:
            results['T'] = calibrator.T[..., tf.newaxis] + results['T'] if self.kind == GSA.Kind.TOTAL else results['T']
            results['T'] = tf.concat([results['T'], calibrator.T[..., tf.newaxis]], axis=-1)
        return results
