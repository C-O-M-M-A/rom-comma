
Getting Started
==========================================================================================================================================

Welcome to the rom-comma python library, which implements Reduction of Order by Marginalization (ROM) via
`Global Sensitivity Analysis (GSA) <https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184>`_
using
`Gaussian Process Regression (GPR) <https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184>`_.
ROM is a novel variance-based approach to `Active Subspaces (AS) <https://epubs.siam.org/doi/book/10.1137/1.9781611973860>`_.
Those unfamiliar with these subjects may wish to follow the hyperlinks in this paragraph to seminal references.

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

User functionality is exposed in the final module, :ref:`romcomma.user`. Any direct call to functionality outside `romcomma.user` is regarded as bespoke, if not
 advanced.



Glossary
---------

.. glossary:

package
    Python package

module
    Python package or module.