"""
This is just the clustering function. It needs to take input variables of all the training data and cluster each hour into clusters.
To do this all the data needs to be standardised.
But the GP only standardises continuous variables. The categorical variables (e.g. day of week do not get standardised.
So the clustering data takes the "UnS_Train_Data.csv" and for each hour cleates a new variable called "Cluster Number".
This "Cluster_Number" is a categorical variable so gets added to the end of "Train_Data.csv"
which has had the continuous variables standardised already and creates a new csv called "Clustered_Data.csv".
"""

from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, is_monotonic    # For cluster analysis
from scipy.cluster.hierarchy import fcluster   # For cluster membership
from matplotlib import pyplot as plt
import tkinter as _tkinter
import math
from pathlib import Path
import time
from numpy import full
import romcomma.model as model
from romcomma.data import Store, Fold, Frame
from romcomma.typing_ import NP, Union, Tuple, Sequence
from shutil import rmtree, copytree
from scipy import optimize
from pyDOE import lhs
from scipy.stats.distributions import norm
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans


def pre_cluster_step(base_path: Union[str, Path], source: str = "UnS_Train_Data.csv", stand_source: str = "Train_Data.csv",
                     clustered_source: str = "Clustered_Data.csv", cutoff: str = "acceleration"):
    """

    Args:
        base_path: The pathway to the file store.
        clustered_source: The name of the csv that the new clustered data will be saved as and will be the source of the following GP.
        source: The initial data used for clustering.
        stand_source: The initial data that has already had it's continuous variables standardised.
        cutoff: A string telling the function which cutoff method to use.
    """
    base_path = Path(base_path)
    df = pd.read_csv(base_path / source, header=[0, 1], index_col=0)  # read the source data file and state the header rows and the index col.
    X = df.copy()  # copy the source file so that the tidying of data that needs clustering doesn't affect the original file.
    X.drop(X.columns[[-1]], axis=1, inplace=True)  # drop the output as that doesn't need clustering

    def standardise(dataset):
        dataNorm = ((dataset - dataset.mean()) / dataset.std())
        # dataNorm[5] = dataset[5] # this may eventually be implemented so that some columns are not standardised.
        return dataNorm

    X = standardise(X)  # standardise the input values using the above function
    input_array = np.array(X)  # ensure inputs to be clustered are now a ndarray, not a dataframe.
    # Generate linkage matrix (this does all work, after this it's just a case of defining cluster cut-off points)
    Z = linkage(input_array, 'ward')
    '''Below we use the fcluster function to get the cluster membership. We must set a cutoff, i.e. a threshold for 
    judging a cluster to be genuinely separate. The fcluster function takes a merge distance as it it's cutoff, i.e. 
    merges with distance >= cutoff have joined two genuine clusters.'''

    def fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Sample index')
            plt.ylabel('Distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata

    # Elbow plot, for guidance on where to truncate dendrogram
    last = Z[-100:, 2]  # Sub array of last 100 merges
    last_rev = last[::-1]  # Reverse the series
    idxs = np.arange(1, len(last) + 1)

    plt.figure(figsize=(25, 10))
    plt.title('Elbow plot')
    plt.xlabel('Clusters identified')
    plt.ylabel('Distance travelled to join clusters')
    plt.plot(idxs, last_rev, label="Dist", marker='D')

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]  # Reverse the series
    plt.plot(idxs[:-2] + 1, acceleration_rev, label="2nd Deriv Dist", marker='.')
    plt.legend()
    plt.show(block=False)

    print("idxs", idxs)
    print("last_rev", last_rev)
    print("accel_rev", acceleration_rev)
    """
    last = Z[-100:, 2] # Sub array of last 100 merges
    last_rev = last[::-1] # Reverse the series
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1] # Reverse the series
    """
    # define cutoff point for intercluster distance
    if cutoff == 'manual':
        print('manual')
        n = int(input('Select number of clusters (i.e. last n-1 merges)'))
        # Select merge distance corresponding to merge that takes us from n clusters to n-1
        cutoff = (last_rev[n-1] + last_rev[n-2])/2 # 0th entry in 'last_rev' is 1 cluster containing all vectors
    elif cutoff == 'acceleration':
        # Pullout index of maximum acceleration
        max_acc = max(acceleration_rev)
        n = list(acceleration_rev).index(max_acc) + 2  # + 2 as 0th entry in acceleration_rev is merge 2 -> 1 clusters
        cutoff = (last_rev[n-1] + last_rev[n-2])/2  # 0th entry in 'last_rev' is 1 cluster containing all vectors
    else:
        cutoff = 1500
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    fancy_dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        no_labels=True,  # Suppress singleton cluster labels that clog up x -axis
        annotate_above=cutoff,  # Prevent excessive annotation of merge distances
        max_d=cutoff)
    plt.show(block=False)

    # Retrieve cluster membership for each vector (hour in our example)
    cluster_membership = fcluster(Z, cutoff, criterion='distance')
    # retrieve the number of clusters
    cluster_list = []
    for i in cluster_membership:
        if i not in cluster_list:
            cluster_list += [i]
    n = len(cluster_list)
    print("number of clusters = ", n)

    # CSV for plotting average profile for each cluster
    y_DF = pd.DataFrame(input_array)
    clusters = pd.DataFrame(cluster_membership)
    clusters_for_means = pd.concat([clusters, y_DF], axis=1)
    cfm_frame_name = "clusters_for_means_cutoff=" + str(cutoff) + "_n=" + str(n) + ".csv"
    clusters_for_means_frame = Frame(base_path / cfm_frame_name, clusters_for_means)
    # clusters_for_means.to_csv("clusters_for_means_cutoff=" + str(cutoff) + "_n=" + str(n) + ".csv", sep=',')
    # get list of cluster membership to CSV for Aaron
    clusters_frame_name = "multi_input_clustering_cutoff=" + str(cutoff) + "_n=" + str(n) + ".csv"
    clusters_frame = Frame(base_path / clusters_frame_name, clusters)
    # clusters.to_csv("multi_input_clustering_cutoff=" + str(cutoff) + "_n=" + str(n) + ".csv", sep=',')
    # Elbow and acceleration plot to CSV
    # Pad acceleration plot so it fits in CSV with others (use large number so it's
    # obvious that they are not part of the real dataset
    acceleration_rev_pad = [1000000]
    for i in range(len(acceleration_rev)):
        acceleration_rev_pad = acceleration_rev_pad + [acceleration_rev[i]]
    acceleration_rev_pad = acceleration_rev_pad + [1000000]
    elbow_plot = pd.DataFrame({"Index": idxs, "Distance to Merge": last_rev, "Acceleration": acceleration_rev_pad})
    ep_frame_name = "elbow_plot_cutoff=" + str(cutoff) + "_n=" + str(n) + ".csv"
    elbow_plot_frame = Frame(base_path / ep_frame_name, elbow_plot)
    # elbow_plot.to_csv("elbow_plot_cutoff=" + str(cutoff) + "_n=" + str(n) + ".csv", sep=',')

    df2 = pd.read_csv(base_path / stand_source, header=[0, 1], index_col=0)
    loc = len(df2.columns) - 1
    df2.insert(loc, "X", cluster_membership)
    df2 = df2.rename({'X': 'Input', '': 'Cluster_No'}, axis=1)
    # now need to save that df as a csv which can be used for GP
    frame = Frame(base_path / clustered_source, df2)
    return frame, elbow_plot_frame, clusters_frame, clusters_for_means_frame


def k_cluster_step(base_path: Union[str, Path], source: str = "UnS_Train_Data.csv", stand_source: str = "Train_Data.csv",
                     clustered_source: str = "Clustered_Data.csv"):
    """

    Args:
        base_path: The pathway to the file store.
        clustered_source: The name of the csv that the new clustered data will be saved as and will be the source of the following GP.
        source: The initial data used for clustering.
        stand_source: The initial data that has already had it's continuous variables standardised.
        cutoff: A string telling the function which cutoff method to use.
    """
    base_path = Path(base_path)
    df = pd.read_csv(base_path / source, header=[0, 1], index_col=0)  # read the source data file and state the header rows and the index col.
    X = df.copy()  # copy the source file so that the tidying of data that needs clustering doesn't affect the original file.
    X.drop(X.columns[[-1]], axis=1, inplace=True)  # drop the output as that doesn't need clustering
    for i in range(5):
        X.drop(X.columns[[-1]], axis=1, inplace=True)  # drop the date cols

    def standardise(dataset):
        dataNorm = ((dataset - dataset.mean()) / dataset.std())
        # dataNorm[5] = dataset[5] # this may eventually be implemented so that some columns are not standardised.
        return dataNorm

    X = standardise(X)  # standardise the input values using the above function
    X = np.array(X)  # ensure inputs to be clustered are now a ndarray, not a dataframe.
    k = 3
    kmc = KMeans(n_clusters=k, init='random', n_init=5, max_iter=300, tol=1e-04, random_state=0)
    kmc.fit_predict(X)
    membership = kmc.labels_
    df2 = pd.read_csv(base_path / stand_source, header=[0, 1], index_col=0)
    loc = len(df2.columns) - 1
    df2.insert(loc, "X", membership)
    df2 = df2.rename({'X': 'Input', '': 'Cluster_No'}, axis=1)
    # now need to save that df as a csv which can be used for GP
    frame = Frame(base_path / clustered_source, df2)
    return frame


if __name__ == '__main__':
    start = time.time()
    Base_Path = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\2019_Testing2\\12-Dec_Trial")
    store_name = "GPc1"
    Source = 'UnS_Train_Data.csv'
    Stand_Source = "UnS_Train_Data.csv"
    Clustered_Source = 'Clustered_Data.csv'
    clustering = k_cluster_step(Base_Path, Source, Stand_Source)
