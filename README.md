# rom-comma

**Gaussian Process Regression, Global Sensitivity Analysis and Reduced Order Modelling by COMMA Research at The University of Sheffield**

## installation
Simply place the `romcomma` package in a folder included in your `PYTHONPATH` (e.g. `site-packages`). 
Test the installation by running the `installation_test` module, from anywhere.

## documentation
Dependencies are documented in `pyproject.toml`.

Full documentation for the `romcomma` package is published on [readthedocs](https://rom-comma.readthedocs.io/en/latest/).

## getting started
The following is not intended to substitute full package documentation, but to sketch the most essential, salient and practically important architectural 
features of the `romcomma` package. These are introduced by module (or package) name, in order of workflow priority. 
Presumably this will reflect the package users' first steps. Familiarity with Gaussian Processes (GPs), Global Sensitivity Analysis (GSA) and 
Reduction of Order by Marginalization (ROM) is largely assumed.

### `data`
The `data` module contains classes for importing and storing the data being analyzed.

Import is from csv file or pandas DataFrame, in any case tabulated with precisely two header rows as

| | Input <br /> _X_<sub>1</sub> | ... <br /> ... | Input <br /> _X_<sub>M</sub> | Output <br /> _Y_<sub>1</sub> | ... <br /> ... | Output <br /> _Y_<sub>L</sub> |
|---| ----- | --- | ----- | ------ | --- | ------ |
| optional column <br /> of _N_ row indices | _N_ rows of <br /> numeric <br /> data |...| _N_ rows of <br /> numeric <br /> data | _N_ rows of <br /> numeric <br /> data |...| _N_ rows of <br /> numeric <br/> data |

Any first-line header may be used instead of "Input", so long as it is the same for every column to be treated as input.
Any first-line header may be used instead of "Output", so long as it is the same for every column to be treated as output, 
and is different to the first-line header for inputs.

Any second-line headers may be used, without restriction. But internally, the `romcomma` package sees
* An (_N_, _M_)  design matrix of inputs called _X_.
* An (_N_, _L_)  design matrix of outputs called _Y_.

The key assumption is that each input column is sampled from a uniform distribution _X_<sub>i</sub> ~ U[_min_<sub>i</sub>, _max_<sub>i</sub>].
There is no claim that the methods used by this software have any validity at all if this assumption is violated. 

In case _X_<sub>_i_</sub> ~ CDF[_X_<sub>i</sub>] the user should apply the probability transform CDF(_X_<sub>i</sub>) ~ U[0, 1] to the input column _i_ 
__prior to any data import__.

#### `Repository`
Data is initially imported into a `Repository` object, which handles storage, retrieval and metadata for `repo.data`.
Every `Repository` object writes to and reads from its own `repo.folder`.

Every `Repository` object crucially exposes a parameter _K_ which triggers 
[k-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation) for this `repo`.
Setting `repo.K=K` generates _K_ `Fold` objects.

#### `Fold`
All data analysis is performed on `Fold` objects. A `Fold` is really a kind of `Repository`, with the addition of
* `fold.test_data`, stored in a table (`Frame`) of _N_/_K_ rows. 
The `test_data` does not overlap the (training) `data` in this `Fold`, except when the parent `repo.K=1` and the ersatz `fold.test_data=fold.data` is applied.
* `Normalization` of inputs: All training and test data inputs are transformed from _X_<sub>i</sub> ~ U[_min_<sub>i</sub>, _max_<sub>i</sub>] 
to the standard normal distribution _X_<sub>_i_</sub> ~ N[0, 1], as demanded by the analyses implemented by `romcomma`.
Outputs are simultaneously normalized to zero mean and unit variance.
`Normalization` exposes an `undo` method to return to the original variables used in the parent `Repository`.

The `repo.K` `Folds` are stored under the parent, in `fold.folder=repo.folder\fold.k` for `k in range(repo.K)`. 
For the purposes of model integration, an unvalidated, ersatz `fold.K` is included with _N_ datapoints of (training) `data=test_data`, 
just like the would-be ersatz _K_=1=_k_+1.
