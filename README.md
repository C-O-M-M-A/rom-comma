# rom-comma
###### Reduced Order Modelling software produced by Solomon F. Brown's COMMA Research Group at The University of Sheffield

## installation
Simply place the `romcomma` package in a folder included in `PYTHONPATH` (e.g. `site-packages`). 
Test the installation by running the `installation_test` module, from anywhere.

## documentation
Dependencies are documented in `pyproject.toml`.

Full documentation for the `romcomma` package is available at https://rom-comma.readthedocs.io/en/latest/.

## getting started
The following is not intended to substitute full package documentation, but to sketch the most essential, salient and practically important architectural 
features of the `romcomma` package. These are introduced by module (or package) name, in order of workflow priority. 
Presumably this will reflect the package users' first steps. Familiarity with Gaussian Processes (GPs), Global Sensitivity Analysis (GSA) and 
Reduction of Order by Marginalization (ROM) is largely assumed.

### `data`
The `data` module contains classes for importing and storing the data being analyzed.

Import is from csv file or pandas DataFrame, in any case tabulated with two header rows as

| Input <br/> X<sub>1</sub> | ... <br/> ... | Input <br/> X<sub>M</sub> | Output <br/> Y<sub>1</sub> | ... <br/> ... | Output <br/> Y<sub>L</sub> |
| ----- | --- | ----- | ------ | --- | ------ |
| numeric <br/> data |...|data|data|...|data|

