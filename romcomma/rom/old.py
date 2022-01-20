#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2022 Robert A. Milton. All rights reserved.
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

""" Contains Reduced Order Modelling tools."""

# # noinspection PyPep8Naming
# class ROM(Model):
#     """ Reduced Order Model (ROM) Calculator and optimizer.
#     This class is documented through its public properties."""
#
#     """ Required overrides."""
#
#     class GP_Initializer(IntEnum):
#         ORIGINAL = auto()
#         ORIGINAL_WITH_CURRENT_KERNEL = auto()
#         ORIGINAL_WITH_GUESSED_LENGTHSCALE = auto()
#         CURRENT = auto()
#         CURRENT_WITH_ORIGINAL_KERNEL = auto()
#         CURRENT_WITH_GUESSED_LENGTHSCALE = auto()
#         RBF = auto()
#
#     MEMORY_LAYOUT = "OVERRIDE_THIS with 'C','F' or 'A' (for C, Fortran or C-unless-All-input-is-Fortran-layout)."
#
#     Parameters = NamedTuple("Parameters", [('Mu', NP.Matrix), ('D', NP.Matrix), ('S1', NP.Matrix), ('S', NP.Matrix),
#                                            ('lengthscales', NP.Matrix), ('log_marginal_likelihood', NP.Matrix)])
#     """
#         **Mu** -- A numpy [[int]] specifying the number of input dimensions in the rotated basis u.
#
#         **D** -- An (L L, M) Matrix of cumulative conditional variances D[l,k,m] = S[l,k,m] D[l,k,M].
#
#         **S1** -- An (L L, M) Matrix of Sobol' main indices.
#
#         **S** -- An (L L, M) Matrix of Sobol' cumulative indices.
#
#         **lengthscales** -- A (1,M) Covector of RBF lengthscales, or a (1,1) RBF lengthscales.
#
#         **log_marginal_likelihood** -- A numpy [[float]] used to record the log marginal likelihood.
#     """
#     PARAMETERS = Parameters(*(atleast_2d(None),) * 6)
#
#     OPTIMIZER_OPTIONS = {'iterations': 1, 'guess_identity_after_iteration': 1, 'sobol_options': Sobol.OPTIMIZER_OPTIONS,
#                                  'gp_initializer': GP_Initializer.CURRENT_WITH_GUESSED_LENGTHSCALE,
#                                  'gp_options': GP.OPTIONS}
#     """
#         **iterations** -- The number of ROM iterations. Each ROM iteration essentially calls Sobol.optimimize(options['sobol_options'])
#             followed by GP.optimize(options['gp_options'])).
#
#         **sobol_options*** -- A Dict of Sobol optimizer options, similar to (and documented in) Sobol.OPTIMIZER_OPTIONS.
#
#         **guess_identity_after_iteration** -- After this many ROM iterations, Sobol.optimize does no exploration,
#             just gradient descending from Theta = Identity Matrix.
#
#         **reuse_original_gp** -- True if GP.optimize is initialized each time from the GP originally provided.
#
#         **gp_options** -- A Dict of GP optimizer options, similar to (and documented in) GP.OPTIMIZER_OPTIONS.
#     """
#
#     @classmethod
#     @abstractmethod
#     def from_ROM(cls, fold: Fold, name: str, suffix: str = ".0", Mu: int = -1, rbf_parameters: Optional[GP.Parameters] = None) -> ROM:
#         """ Create a ROM object from a saved ROM folder.
#
#         Args:
#             fold: The Fold housing the ROM to load.
#             name: The name of the saved ROM to create from.
#             suffix: The suffix to append to the most optimized gp.
#             Mu: The dimensionality of the rotated input basis u. If this is not in range(1, fold.M+1), Mu=fold.M is used.
#
#         Returns: The constructed ROM object
#         """
#         optimization_count = [optimized.name.count(cls.OPTIMIZED_GB_EXT) for optimized in fold.folder.glob("name" + cls.OPTIMIZED_GB_EXT + "*")]
#         source_gp_name = name + cls.OPTIMIZED_GB_EXT * max(optimization_count)
#         destination_gp_name = source_gp_name + suffix
#         return cls(name=name,
#                    sobol=Sobol.from_GP(fold, source_gp_name, destination_gp_name, Mu=Mu, read_parameters=True),
#                    options=None, rbf_parameters=rbf_parameters)
#
#     @classmethod
#     @abstractmethod
#     def from_GP(cls, fold: Fold, name: str, source_gp_name: str, options: Dict, Mu: int = -1,
#                 rbf_parameters: Optional[GP.Parameters] = None) -> ROM:
#         """ Create a ROM object from a saved GP folder.
#
#         Args:
#             fold: The Fold housing the ROM to load.
#             name: The name of the saved ROM to create from.
#             source_gp_name: The source GP folder.
#             Mu: The dimensionality of the rotated input basis u. If this is not in range(1, fold.M+1), Mu=fold.M is used.
#             options: A Dict of ROM optimizer options.
#
#         Returns: The constructed ROM object
#         """
#         return cls(name=name,
#                    sobol=Sobol.from_GP(fold=fold, source_gp_name=source_gp_name, destination_gp_name=name + ".0", Mu=Mu),
#                    options=options, rbf_parameters=rbf_parameters)
#
#     OPTIMIZED_GP_EXT = ".optimized"
#     REDUCED_FOLD_EXT = ".reduced"
#
#     """ End of required overrides."""
#
#     @property
#     def name(self) -> str:
#         """ The name of this ROM."""
#         return self.folder.name
#
#     @property
#     def sobol(self) -> Sobol:
#         """ The Sobol object underpinning this ROM."""
#         return self._sobol
#
#     @property
#     def gp(self) -> Sobol:
#         """ The GP underpinning this ROM."""
#         return self._gp
#
#     @property
#     def semi_norm(self) -> Sobol.SemiNorm:
#         """ A Sobol.SemiNorm on the (L,L) matrix of Sobol' indices, defining the ROM optimization objective ``semi_norm(D[:,:,m])``."""
#         return self._semi_norm
#
#     def gp_name(self, iteration: int) -> str:
#         """ The name of the GP produced by iteration."""
#         if iteration >= 0:
#             return "{0}.{1:d}".format(self.name, iteration)
#         else:
#             return "{0}{1}".format(self.name, self.OPTIMIZED_GB_EXT)
#
#     def _initialize_gp(self, iteration: int) -> GP:
#         if self._rbf_parameters is not None:
#             gp_initializer = self.GP_Initializer.RBF
#             parameters = self._rbf_parameters
#             gp_rbf = self.GPType(self._fold, self.gp_name(iteration) + ".rbf", parameters)
#             gp_rbf.optimize(**self._options[-1]['gp_options'])
#             gp_dir = gp_rbf.folder.parent / self.gp_name(iteration)
#             Model.copy(gp_rbf.folder, gp_dir)
#             kernel = type(self._gp.kernel)(None, None, gp_dir / GP.KERNEL_DIR_NAME)
#             kernel.make_ard(self._gp.M)
#             return self.GPType(self._fold, self.gp_name(iteration), parameters=None)
#         gp_initializer = self._options[-1]['gp_initializer']
#         parameters = self._original_parameters if gp_initializer < self.GP_Initializer.CURRENT else self._gp.parameters
#         if not self._gp.kernel.is_rbf:
#             if gp_initializer in (self.GP_Initializer.ORIGINAL_WITH_GUESSED_LENGTHSCALE, self.GP_Initializer.CURRENT_WITH_GUESSED_LENGTHSCALE):
#                 lengthscales = einsum('MK, JK -> M', self._sobol.Theta_old, self._gp.kernel.parameters.lengthscales, optimize=True, dtype=float,
#                                       order=self.MEMORY_LAYOUT) * 0.5 * self._gp.M * (self._gp.M - arange(self._gp.M, dtype=float)) ** (-1)
#             elif gp_initializer in (self.GP_Initializer.CURRENT_WITH_ORIGINAL_KERNEL, self.GP_Initializer.ORIGINAL):
#                 lengthscales = einsum('MK, JK -> M', self._Theta, self._original_parameters.kernel.parameters.lengthscales,
#                                       optimize=True, dtype=float, order=self.MEMORY_LAYOUT)
#             elif gp_initializer in (self.GP_Initializer.ORIGINAL_WITH_CURRENT_KERNEL, self.GP_Initializer.CURRENT):
#                 lengthscales = einsum('MK, JK -> M', self._sobol.Theta_old, self._gp.kernel.parameters.lengthscales, optimize=True, dtype=float,
#                                       order=self.MEMORY_LAYOUT)
#             parameters = parameters._replace(kernel=self._gp.kernel.Parameters(lengthscales=lengthscales))
#         return self.GPType(self._fold, self.gp_name(iteration), parameters)
#
#     def optimize(self, options: Dict):
#         """ Optimize the model parameters. Do not call super().optimize, this interface only contains suggestions for implementation.
#
#         Args:
#             options: A Dict of implementation-dependent optimizer options, following the format of ROM.OPTIMIZER_OPTIONS.
#         """
#         if options is not self._options[-1]:
#             self._options.append(options)
#             self._semi_norm = Sobol.SemiNorm.from_meta(self._options[-1]['sobol_options']['semi_norm'])
#             self._sobol_reordering_options['semi_norm'] = self._semi_norm
#
#         self._options[-1]['sobol_options']['semi_norm'] = self._semi_norm.meta
#         self._write_options(self._options)
#
#         iterations = self._options[-1]['iterations']
#         if iterations < 1 or self._options[-1]['sobol_options']['N_exploit'] < 1:
#             if not iterations <= 1:
#                 warn("Your ROM optimization does not allow_rotation so iterations is set to 1, instead of {0:d}.".format(iterations), UserWarning)
#             iterations = 1
#
#         guess_identity_after_iteration = self._options[-1]['guess_identity_after_iteration']
#         if guess_identity_after_iteration < 0:
#             guess_identity_after_iteration = iterations
#
#         sobol_guess_identity = {**self._options[-1]['sobol_options'], 'N_explore': 1}
#         self._Theta = self._sobol.Theta_old
#
#         for iteration in range(iterations):
#             self._gp = self._initialize_gp(iteration + 1)
#             self.calculate()
#             self.write_parameters(self.Parameters(
#                 concatenate((self.parameters.Mu, atleast_2d(self._sobol.Mu)), axis=0),
#                 concatenate((self.parameters.D, atleast_2d(self._semi_norm.ordinate(self._sobol.D))), axis=0),
#                 concatenate((self.parameters.S1, atleast_2d(self._semi_norm.ordinate(self._sobol.S1))), axis=0),
#                 concatenate((self.parameters.S, atleast_2d(self._semi_norm.ordinate(self._sobol.S))), axis=0),
#                 concatenate((self.parameters.lengthscales, atleast_2d(self._sobol.lengthscales)), axis=0),
#                 concatenate((self.parameters.log_marginal_likelihood, atleast_2d(self._gp.log_marginal_likelihood)), axis=0)))
#             if iteration < guess_identity_after_iteration:
#                 self._sobol.optimize(**self._options[-1]['sobol_options'])
#             else:
#                 self._sobol.optimize(**sobol_guess_identity)
#             self._Theta = einsum('MK, KL -> ML', self._sobol.Theta_old, self._Theta)
#
#         self._gp = self._initialize_gp(-1)
#         self.calculate()
#         self._gp.test_data()
#         self.write_parameters(self.Parameters(
#             concatenate((self.parameters.Mu, atleast_2d(self._sobol.Mu)), axis=0),
#             concatenate((self.parameters.D, atleast_2d(self._semi_norm.ordinate(self._sobol.D))), axis=0),
#             concatenate((self.parameters.S1, atleast_2d(self._semi_norm.ordinate(self._sobol.S1))), axis=0),
#             concatenate((self.parameters.S, atleast_2d(self._semi_norm.ordinate(self._sobol.S))), axis=0),
#             concatenate((self.parameters.lengthscales, atleast_2d(self._sobol.lengthscales)), axis=0),
#             concatenate((self.parameters.log_marginal_likelihood, atleast_2d(self._gp.log_marginal_likelihood)), axis=0)))
#         column_headings = ("x{:d}".format(i) for i in range(self._sobol.Mu))
#         frame = Frame(self._sobol.parameters_csv.Theta, DataFrame(self._Theta, columns=column_headings))
#         frame.write()
#
#     def reduce(self, Mu: int = -1):
#         """
#
#         Args:
#             Mu: The reduced dimensionality Mu &le sobol.Mu. If Mu &le 0, then Mu = sobol.Mu.
#
#         Returns:
#         """
#
#     def calculate(self):
#         """ Calculate the Model. """
#         self._gp.optimize(**self._options[-1]['gp_options'])
#         self._sobol = self.SobolType(self._gp)
#
#     def __init__(self, name: str, sobol: Sobol, options: Dict = OPTIMIZER_OPTIONS,
#                  rbf_parameters: Optional[GP.Parameters] = None):
#         """ Initialize ROM object.
#
#         Args:
#             sobol: The Sobol object to construct the ROM from.
#             options: A List[Dict] similar to (and documented in) ROM.OPTIMIZER_OPTIONS.
#         """
#         self._rbf_parameters = rbf_parameters
#         self._sobol = sobol
#         self._gp = sobol.gp
#         self._original_parameters = self._gp.parameters._replace(kernel=self._gp.kernel.parameters)
#         self._sobol_reordering_options = deepcopy(Sobol.OPTIMIZER_OPTIONS)
#         self._fold = Fold(self._gp.fold.folder.parent, self._gp.fold.meta['k'], self._sobol.Mu)
#         self.SobolType = deepcopy(type(self._sobol))
#         self.GPType = deepcopy(type(self._gp))
#         if options is None:
#             super().__init__(self._fold.folder / name, None)
#             self._options = self._read_options()
#         else:
#             self._options = [options]
#             self._semi_norm = Sobol.SemiNorm.from_meta(self._options[-1]['sobol_options']['semi_norm'])
#             self._sobol_reordering_options['semi_norm'] = self._semi_norm
#             parameters = self.Parameters(Mu=self._sobol.Mu,
#                                          D=self._semi_norm.ordinate(self._sobol.D),
#                                          S1=self._semi_norm.ordinate(self._sobol.S1),
#                                          S=self._semi_norm.ordinate(self._sobol.S),
#                                          lengthscales=self._sobol.lengthscales,
#                                          log_marginal_likelihood=self._gp.log_marginal_likelihood)
#             super().__init__(self._fold.folder / name, parameters)
#             shutil.copy2(self._fold.csv, self.folder)
#             shutil.copy2(self._fold._test_csv, self.folder)
#             self.optimize(self._options[-1])
