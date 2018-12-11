""" pca_reduced_clustering.py
This script will cluster the single cell data provided for the project after 
    reducing the dimensionality via PCA.
Arguments: None
Output: Cluster labels written to the cluster_results directory
Author: Ben Heil
Date: 12/11/18
"""

import sklearn.decomposition 
from sklearn.cluster import SpectralClustering, KMeans
import numpy

def run_pca(dataset, num_components):
    """run_pca
    Reduces the dimensionality of the dataset with principle component analysis

    Arguments: dataset - (array_like) The data to be reduced
               num_components - (int) The number of principle components to keep
    Returns: transformed_dataset - (array_like) The reduced dataset
    """

    pca_runner = sklearn.decomposition.PCA(n_components=num_components, 
                                           random_state=42)

    transformed_dataset = pca_runner.fit_transform(dataset)

    return transformed_dataset

def spectral_clustering(dataset, num_clusters):
    """spectral_clustering
    Performs spectral clustering on the provided dataset

    Arguments: dataset - (array_like) Data for which labels should be predicted
               num_clusters - The number of clusters to look for in the data
    Returns: labels - The predicted labels for the dataset
    """
    spectral_clustering_runner = SpectralClustering(n_clusters=num_clusters, 
                                                    random_state=42, n_jobs=3)

    labels = spectral_clustering_runner.fit_predict(dataset)

    return labels
    
def kMeans(dataset, num_clusters):
    """kMeans
    Performs k-means clustering on the provided dataset

    Arguments: dataset - (array_like) Data for which labels should be predicted
               num_clusters - The number of clusters to look for in the data
    Returns: labels - The predicted labels for the dataset
    """
    kMeans_runner = KMeans(n_clusters=num_clusters, random_state=42, n_jobs=3)

    labels = kMeans_runner.fit_predict(dataset)

    return labels

def main():
    print("Reading data file...")
    dataset = numpy.genfromtxt('./data/CelegansRawCounts.csv', delimiter=',')

    print("Running pca...")
    # We use 15 components for this dataset because they caputer ~ 90 percent 
    # of the variance. This is arbitrary and you can pick whatever value you
    # feel would work best (or better yet use cross validation to pick)
    dataset_pcs = run_pca(dataset, 15)

    out_file_name_base = './cluster_results/pca_kmeans_labels_c_X.csv'
    for i in range(2,201):
        print("Running k-means for " + str(i) + " clusters")
        kmeans_labels = kMeans(dataset_pcs, i)

        out_file_name = out_file_name_base.replace('X', str(i))
        numpy.savetxt(out_file_name, kmeans_labels, fmt='%d')

    # Didn't end up running spectral clustering as the PCA reduced data
    # caused it to crash
    

if __name__ == "__main__":
    main()
