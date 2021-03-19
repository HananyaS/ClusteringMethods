import pandas as pd
import random as rnd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing as pp
import matplotlib.pyplot as plt


# get the first data set as a data frame to cluster and a series which is used as the tag
def get_first_df():
    names = [chr(i + ord('A')) for i in range(9)]
    df = pd.read_csv("HTRU_2.csv", names=names)
    # df = pd.read_csv("Datasets/HTRU_2.csv", names=names)
    # separate the data into to cluster and tag
    tag_fields = ['I']
    tag = df[tag_fields]
    to_cluster = df.drop(tag_fields, axis=1)
    return to_cluster, tag


# get the second data set as a data frame to cluster and a series which is used as the tag
def get_second_df():
    df = pd.read_csv("allUsers.lcl.csv")
    # df = pd.read_csv("Datasets/allUsers.lcl.csv")

    # convert the question marks into the median of the column
    convert_question_marks(df)

    # separate the data into to cluster and tag
    tag_fields = ['Class']
    tag = df[tag_fields]
    to_cluster = df.drop(tag_fields, axis=1)
    return to_cluster, tag


# perform a PCA for reducing the dimension of the data to 2
def PCA_alg(df):
    # get the data frame (the one we would like to cluster)
    df_values = df.values
    # before performing the PCA, normalize the values using standard scaler
    normalized_df_values = pp.StandardScaler().fit_transform(df_values)
    # create a PCA object, which reduces data into 2 dimensions
    pca = PCA(n_components=2)
    # do the PCA on the data and return the new reduced columns data frame with the columns names 'PC1' and 'PC2'
    pc = pca.fit_transform(normalized_df_values)
    return pd.DataFrame(data=pc, columns=['PC1', 'PC2'])


# calculate the median of the non-question marks values of an array
def calc_median(col):
    return np.median([float(col.values[i]) for i in range(len(col)) if col.values[i] != '?'])


# convert the question marks into the median of the column
def convert_question_marks(df):
    for col in df.columns:
        # for each column, calculate the median and replace all the question marks with it
        med = calc_median(df[col])
        df[col] = df[col].replace(['?'], med)


# get a data frames for the clustering and for the tag according to the data set number (valid values - 1 to 3 -
# otherwise, throw an exception) if needed, the 'n_samples' parameter let you choose exact number of samples from the
# data set, randomly (its default value is None, means to return the whole data frame)
def get_data(dataset_num, n_samples=None):
    if dataset_num == 1:
        data = get_first_df()
    elif dataset_num == 2:
        data = get_second_df()
    else:
        raise Exception("Dataset number is between 1 to 2")

    # samples_indices = range(1, len(data[0]))

    if n_samples is None:
        n_samples = len(data[0].values)

    samples_indices = rnd.sample(range(len(data[0])), n_samples)

    to_cluster, tag = data
    data = [to_cluster.loc[samples_indices], tag.loc[samples_indices]]
    data[0] = PCA_alg(data[0])
    return data


# get the data frame for the clustering (after PCA)
def get_df_to_cluster(data):
    return data[0]


# get the data frame which is used as a tag
def get_tag(data):
    return data[1]


# get the "real" labels of the data
def get_labels(tag):
    labels = []
    already_converted = {}
    curr_cluster_num = 0
    for r in tag.values:
        if str(r) not in already_converted.keys():
            already_converted[str(r)] = curr_cluster_num
            curr_cluster_num += 1
        labels.append(already_converted[str(r)])
    return labels


# get the "real" number of clusters for the data
def get_num_of_clusters(tag):
    return len(np.unique(get_labels(tag)))


def plot_external_classification(dataset_num):
    data = get_data(dataset_num)
    df = get_df_to_cluster(data)
    labels = get_labels(get_tag(data))

    plt.scatter(df['PC1'], df['PC2'], c=labels)
    title = 'DS {} - External Classification'.format(dataset_num)
    plt.title(title)
    # fig_name = 'Images\External Classification\\' + title
    # plt.savefig(fig_name)
    plt.show()
