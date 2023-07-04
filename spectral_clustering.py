import data_frames as d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_samples


# perform spectral clustering for the data set which its number is stored in 'dataset_num', with 'num_of_clusters'
# clusters
def spectral_clustering_alg(dataset_num, num_of_samples=None, num_of_clusters=None, get_figure=False,
                            calc_ami_score=True, plot_anomaly=False):
    if plot_anomaly and (num_of_samples is None or (num_of_samples is not None and num_of_samples > 6000)):
        num_of_samples = 6000

    data = d.get_data(dataset_num, n_samples=num_of_samples)
    # store the data frame which is given by a dimension reduction (using PCA) into 2 dimensions of the original
    # data set
    df = d.get_df_to_cluster(data)
    tag = d.get_tag(data)

    # if the number of clusters isn't defined, choose it to be the "real" number of clusters (according to the tag)
    if num_of_clusters is None:
        num_of_clusters = d.get_num_of_clusters(tag)

    # create a SpectralClustering-type object with the relevant number of clusters and fit it to the data frame
    sc = SpectralClustering(n_clusters=num_of_clusters, affinity='nearest_neighbors')
    # store the labels result after the fitting
    labels = sc.fit_predict(df)

    if get_figure or plot_anomaly:
        if plot_anomaly:
            silhouettes = silhouette_samples(df, labels)
            labels[silhouettes < 0] = -1

        # calculate the centroids of each cluster
        centroids = [[np.mean(df.values[labels == i][:, 0]), np.mean(df.values[labels == i][:, 1])] for i
                     in range(num_of_clusters)]

        # plot the clustered data and the centroid of each cluster
        plt.scatter(df['PC1'][labels != -1], df['PC2'][labels != -1], c=labels[labels != -1])
        plt.scatter(df['PC1'][labels == -1], df['PC2'][labels == -1], c=['black'] * len(labels[labels == -1]),
                    label='Anomaly')
        plt.scatter([row[0] for row in centroids], [row[1] for row in centroids], c='black', marker='*',
                    label='centroid')
        title = 'DS{} - Spectral Clustering'.format(dataset_num)
        # fig_name = 'Images\Spectral\\' + title
        plt.title(title)
        plt.legend()

        # save the figure
        # plt.savefig(fig_name)
        plt.show()

    # calculate the adjusted mutual info score of the clustering
    if calc_ami_score:
        labels_true = d.get_labels(tag)
        return adjusted_mutual_info_score(labels_true=labels_true, labels_pred=labels)

