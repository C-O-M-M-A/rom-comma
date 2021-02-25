import numpy as np
from romcomma.data import Store
from pathlib import Path
from matplotlib import pyplot as plt


def load_data(source: str, store_name: str, origin_csv_parameters=None) -> Store:
    if origin_csv_parameters is None:
        origin_csv_parameters = {'index_col': 0}
    store = Store.from_csv(BASE_PATH / store_name, BASE_PATH / source, **origin_csv_parameters)
    # from here X is a dataframe of inputs and Y is a dataframe of outputs.
    return store


def plot_data(store: Store, store_name: str, hist_num_name: str = "Data_Num_Hists",
              hist_str_name: str = "Data_Str_Hists_"):
    df = store.Y
    for y in df.columns:
        if df[y].dtype == np.float64 or df[y].dtype == np.int64:
            # df.hist(sharey=True, bins=40)
            df.hist(bins=40)
            plt.tight_layout()
            fig_name = hist_num_name + ".pdf"
            plt.savefig(BASE_PATH / store_name / fig_name)
            plt.clf()
        else:
            df[y].value_counts().plot(kind="bar")
            fig_name = hist_str_name + str(y) + ".pdf"
            plt.savefig(BASE_PATH / store_name / fig_name)
    return


BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Toy_Data_2")
STORE_NAME = "Toy_Data_Dist"
# Store = load_data("Data.csv", STORE_NAME)
Store = Store(BASE_PATH / STORE_NAME)
plot_data(Store, STORE_NAME, "Data_Output_Hists")
