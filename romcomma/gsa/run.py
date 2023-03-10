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

# Contains classes to calculate and record a variety of Global Sensitivity Analyses (GSAs).

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.base.classes import Model, Parameters
from romcomma.gpr.models import GPInterface
from romcomma.gsa import sobol
from enum import Enum, auto


class calculation(Model):
    """ Class encapsulating a general GSA, with calculation and recording facilities."""

    class Parameters(Parameters):
        """ The Parameters set of a GSA."""

        @classmethod
        @property
        def Values(cls) -> Type[NamedTuple]:
            """ The NamedTuple underpinning this Parameters set."""

            class Values(NamedTuple):
                """ The parameters set of a GSA.

                    Attributes:
                        S (NP.Matrix): The Sobol index.
                        T (NP.Matrix): The Sobol index variance.
                        V (NP.Matrix): The conditional variances underpinning the Sobol index.
                        W (NP.Matrix): The covariances underpinning the Sobol index variance.
                """
                S: NP.Matrix = np.atleast_2d(None)
                T: NP.Matrix = np.atleast_2d(None)
                V: NP.Matrix = np.atleast_2d(None)
                W: NP.Matrix = np.atleast_2d(None)

            return Values

    class Kind(Enum):
        """ Enum to specify the kind of Sobol index to calculate."""
        FIRST_ORDER = auto()
        CLOSED = auto()
        TOTAL = auto()

    @classmethod
    @property
    def ALL_KINDS(cls):
        return [kind for kind in cls.Kind]

    @classmethod
    @property
    def OPTIONS(cls) -> Dict[str, Any]:
        """ Default calculation options. ``is_T_partial`` forces ``WmM = 0``."""
        return sobol.ClosedIndexWithErrors.OPTIONS

    @classmethod
    def _calculate(cls, kind: calculation.Kind, m_dataset: tf.data.Dataset, calculate: sobol.ClosedIndex) -> Dict[str, TF.Tensor]:
        """ Perform the GSA calculation.

        Args:
            kind: The Kind of GSA to perform.
            m_dataset: The tf.Dataset of slices to iterate through.
            calculate: The ``calculate.ClosedIndex`` object to do the calculations.
        Returns: The results of the calculation, as a labelled dictionary of tf.Tensors.
        """
        first_iteration = True
        for m in m_dataset:
            result = calculate.marginalize(m)
            if first_iteration:
                results = {key: value[..., tf.newaxis] for key, value in result.items()}
                first_iteration = False
            else:
                for key in results.keys():
                    results[key] = tf.concat([results[key], result[key][..., tf.newaxis]], axis=-1)
        results['V'] = tf.concat([results['V'], calculate.V[0][..., tf.newaxis]], axis=-1)
        results['S'] = calculate.S[..., tf.newaxis] - results['S'] if kind == calculation.Kind.TOTAL else results['S']
        results['S'] = tf.concat([results['S'], calculate.S[..., tf.newaxis]], axis=-1)
        if 'T' in results:
            TT = results.pop('TT')
            results['T'] = TT if (kind == calculation.Kind.TOTAL) else results['T']
        return results

    @classmethod
    def _compose_and_save(cls, csv: Path, value: TF.Tensor, m: int, M: int):
        """ Compose and Save a GSA results tf.Tensor.

        Args:
            csv: The path to save to.
            value: The tf.Tensor of results to compose and save.
            m: The dimensionality of the reduced model.
            M: The dimensionality of the full model.
        """
        m_list = list(range(M)) if m < 0 else [m]
        shape = value.shape.as_list()
        df = pd.DataFrame(tf.reshape(value, [-1, shape[-1]]).numpy(), columns=cls._columns(M, shape[-1], m_list), index=cls._index(shape))
        df.to_csv(csv, float_format='%.6f')

    @classmethod
    def _columns(cls, M: int, m_cols: int, m_list: List[int]) -> pd.Index:
        """ Index pd.DataFrame columns for output.

        Args:
            M: The dimensionality of the full model.
            m_cols: The number of input columns covering ``m``.
            m_list: The list of input columns covering ``m``.
        Returns: A pd.Index of column labels (integers).

        """
        if m_cols > len(m_list):
            m_list = m_list + [M]
        if m_cols > len(m_list):
            m_list = [-1] + m_list
        return pd.Index(m_list, name='m')

    @classmethod
    def _index(cls, shape: List[int]) -> pd.MultiIndex:
        """ Index a pd.DataFrame for output.

        Args:
            shape: The shape of the pd.DataFrame to index.
        Returns: The pd.MultiIndex to index the pd.DataFrame
        """
        shape = shape[:-1]
        indices = [list(range(l)) for l in shape]
        columns = len(shape)
        return pd.MultiIndex.from_product(indices, names=[f'l.{l}' for l in range(len(indices))])

    def _m_dataset(self, kind: calculation.Kind, m: int, M: int) -> tf.data.Dataset:
        """ ``m`` as a tf.Dataset for iteration.

        Args:
            kind: A GSA.Kind specifying the kind of GSA.
            m: The final index of the reduced model.
            M: The dimensionality of the full model.
        Returns: A dataset of slices to iterate through to perform th GSA.
        """
        result = []
        ms = range(M) if m < 0 else [m]
        if kind == calculation.Kind.FIRST_ORDER:
            result = [tf.constant([m, m + 1], dtype=INT()) for m in ms]
        elif kind == calculation.Kind.CLOSED:
            result = [tf.constant([0, m + 1], dtype=INT()) for m in ms]
        elif kind == calculation.Kind.TOTAL:
            result = [tf.constant([m + 1, M], dtype=INT()) for m in ms]
        return tf.data.Dataset.from_tensor_slices(result)

    def __repr__(self) -> str:
        return str(self.folder)

    def __str__(self) -> str:
        return self.folder.name

    def __init__(self, gp: GPInterface, kind: calculation.Kind, m: int = -1, is_error_calculated: bool = False, **kwargs: Any):
        """ Perform a Sobol GSA. The object created is single use and disposable: the constructor performs and records the entire GSA and the
        constructed object is basically useless once constructed.

        Args:
            gp: The underlying Gaussian Process.
            kind: The kind of index to calculate - first order, closed or total.
            m: The final index of the reduced model. For a single calculation it is required that ``0 &le m &lt gp.M``.
                Any m outside this range results the Sobol index of kind being calculated for all ``m in range(1, M+1)``.
            is_error_calculated: Whether to calculate the standard error on the Sobol index.
                This is a memory intensive calculation, so leave this flag False unless you are sure you need errors
            **kwargs: The calculation options to override OPTIONS.
        """
        m, name = (m, f'{kind.name.lower()}.{m}') if 0 <= m < gp.M else (-1, kind.name.lower())
        options = {'m': m} | self.OPTIONS | kwargs
        folder = gp.folder / 'gsa' / name
        super().__init__(folder, read_parameters=False)
        self._write_options(options)
        results = self._calculate(kind=kind, m_dataset=self._m_dataset(kind, m, gp.M),
                                  calculate=sobol.ClosedIndexWithErrors(gp, **options) if is_error_calculated else sobol.ClosedIndex(gp, **options))
        for key, value in results.items():
            self._compose_and_save(self.parameters.csv(key), value, m, gp.M)
