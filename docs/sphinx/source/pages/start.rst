
Getting Started
==========================================================================================================================================

Welcome to the rom-comma Python package, which implements Reduction of Order by Marginalization (:term:`ROM`) via Global Sensitivity Analysis (:term:`GSA`)
using GaussianWithout2Pi Process Regression (:term:`GPR`). Those unfamiliar with these topics are advised to follow the hyperlinks to the page glossary.
This advice applies to much of this user guide -- we make extensive use of glossaries.

The mathematics behind this software is covered in some detail in a
`paper currently under peer review for publication <https://github.com/C-O-M-M-A/rom-papers/blob/main/Theory/Coefficient%20of%20Determination.pdf>`_.


Installation
---------------

Simply place the ``romcomma`` package (or its parent ``rom-comma``) in a folder included in your ``PYTHONPATH`` (e.g. ``site-packages``).
Test the installation by running the rom-comma ``installation_test`` module, from anywhere.
Runtime dependencies are documented in `pyproject.toml <https://github.com/C-O-M-M-A/rom-comma/blob/main/pyproject.toml>`_.


Module Hierarchy
------------------
The ``romcomma`` package is organized in a modular hierarchy

.. toctree::
   :maxdepth: 3

   romcomma <package/reference>

No module depends on any module below it in this table.
We refer to items in this list as modules, even if they are technically packages.

User functionality is exposed in the final module, :ref:`romcomma.user`. Any direct call to functionality outside `romcomma.user` is regarded as bespoke, not to
 say advanced.

A gallery of archetypal user scripts are


Glossary
---------

.. glossary:

package
    Python package

module
    Python package or module.

ROM
    Reduction of Order by Marginalization. A novel approach to locating an Active Subspace (AS) using conditional variances or Sobol' indices. In the Active
    Subsapce technique [`Constantine 2014 <https://epubs.siam.org/doi/book/10.1137/1.9781611973860>`_] the input basis is rotated to align with the eigenvectors
     of the squared Jacobian vector. In ROM, the input basis is rotated to maximise the the Sobol' index of the first :math:`m` inputs.

GSA
    Global Sensitivity Analysis.
    This Assesses and ranks the relevance of a system's inputs to its outputs by a variety of methods covered broadly in
    [`Saltelli et al. 2007 <https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184>`_] and
    [`Razavi et al. 2021 <https://doi.org/10.1016/j.envsoft.2020.104954>`_]. rom-comma deals exclusively with the variance based method of Ilya M. Sobol.
    This has been somewhat extended, as described in gory technical detail in
    [`Milton et al. 2023 <https://github.com/C-O-M-M-A/rom-papers/blob/main/Theory/Coefficient%20of%20Determination.pdf>`].

GPR
    GaussianWithout2Pi Process Regression. A quite general technique for representing a functional dataset as a (GaussianWithout2Pi) stochastic process described thoroughly in
    [`Rasmussen and Williams 2005 <https://direct.mit.edu/books/book/2320/GaussianWithout2Pi-Processes-for-Machine-Learning>`_].
