from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import Data_frames as d
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np


# perform DBSCAN clustering for the data set which its number is stored in 'dataset_num', with 'num_of_clusters'
# clusters
def dbscan_alg(dataset_num, num_of_samples=None, get_figure=False, calc_ami_score=True, eps=.25, min_samples=5):
    if dataset_num == 2 and num_of_samples is None:
        num_of_samples = 60000
    data = d.get_data(dataset_num, n_samples=num_of_samples)

    # store the data frame which is given by a dimension reduction (using PCA) into 2 dimensions of the original
    # data set
    df = d.get_df_to_cluster(data)
    tag = d.get_tag(data)

    # create a GaussianMixture-type object with the relevant number of clusters and fit it to the data frame
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(df)
    # store the labels result after the fitting
    labels = dbscan.labels_
    if get_figure:
        # calculate the centroids of each cluster
        centroids = [[np.mean(df.values[labels == i][:, 0]), np.mean(df.values[labels == i][:, 1])] for i
                     in range(len(labels) - 1)]

        # plot the clustered data and the centroid of each cluster
        plt.scatter(df['PC1'], df['PC2'], c=labels)
        plt.scatter(df.values[labels == -1][:, 0], df.values[labels == -1][:, 1], c='black', label='Anomaly')

        plt.scatter([row[0] for row in centroids], [row[1] for row in centroids], c='black', marker='*',
                    label='centroid')

        title = 'DS{} - DBSCAN'.format(dataset_num)
        # fig_name = 'Images\DBSCAN\\' + title
        plt.title(title)
        plt.legend()

        # save the figure
        # plt.savefig(fig_name)
        plt.show()

    # calculate the adjusted mutual info score of the clustering
    if calc_ami_score:
        labels_true = d.get_labels(tag)
        return adjusted_mutual_info_score(labels_true=labels_true, labels_pred=labels)


dbscan_alg(1, get_figure=True)
dbscan_alg(2, get_figure=True)