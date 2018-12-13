""" tsne_first_clustering.py
This script will cluster the single cell data provided for the project after 
    reducing the dimensionality via tsne
Arguments: None
Output: Cluster labels written to the cluster_results directory
Author: Ben Heil
Date: 12/11/18
"""

import sklearn.decomposition 
import sklearn.manifold
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

def run_lle(dataset, num_components):
    """run_tsne
    Reduces the dimensionality of the dataset with t-Stochastic neighbor embedding 

    Arguments: dataset - (array_like) The data to be reduced
               num_components - (int) The number of dimensions to return 
    Returns: transformed_dataset - (array_like) The reduced dataset
    """

    tsne_runner = sklearn.manifold.LocallyLinearEmbedding(n_components=num_components, 
                                           random_state=42, n_jobs=4)

    transformed_dataset = tsne_runner.fit_transform(dataset)

    return transformed_dataset

def run_tsne(dataset, num_components):
    """run_tsne
    Reduces the dimensionality of the dataset with t-Stochastic neighbor embedding 

    Arguments: dataset - (array_like) The data to be reduced
               num_components - (int) The number of dimensions to return 
    Returns: transformed_dataset - (array_like) The reduced dataset
    """

    tsne_runner = sklearn.manifold.TSNE(n_components=num_components, 
                                           random_state=42)

    transformed_dataset = tsne_runner.fit_transform(dataset)

    return transformed_dataset

def spectral_clustering(dataset, num_clusters):
    """spectral_clustering
    Performs spectral clustering on the provided dataset

    Arguments: dataset - (array_like) Data for which labels should be predicted
               num_clusters - The number of clusters to look for in the data
    Returns: labels - The predicted labels for the dataset
    """
    spectral_clustering_runner = SpectralClustering(n_clusters=num_clusters, 
                                                    random_state=42, n_jobs=4)

    labels = spectral_clustering_runner.fit_predict(dataset)

    return labels
    
def kMeans(dataset, num_clusters):
    """kMeans
    Performs k-means clustering on the provided dataset

    Arguments: dataset - (array_like) Data for which labels should be predicted
               num_clusters - The number of clusters to look for in the data
    Returns: labels - The predicted labels for the dataset
    """
    kMeans_runner = KMeans(n_clusters=num_clusters, random_state=42, n_jobs=4)

    labels = kMeans_runner.fit_predict(dataset)

    return labels

def main():
    print("Reading data file...")
    dataset = numpy.genfromtxt('./data/CelegansRawCounts.csv', delimiter=',')
    dataset += 1
    dataset = numpy.log(dataset)

    print("Running tsne...")
    dataset_pcs = run_lle(dataset, 3)

    out_file_name_base = './cluster_results/log_lle_kmeans_labels_c_X.csv'
    for i in range(2,201):
        print("Running k-means for " + str(i) + " clusters")
        kmeans_labels = kMeans(dataset_pcs, i)

        out_file_name = out_file_name_base.replace('X', str(i))
        numpy.savetxt(out_file_name, kmeans_labels, fmt='%d')
    

if __name__ == "__main__":
    main()
