from sklearn.cluster import KMeans
from preprocessing import toDF
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

import numpy as np
from scipy.stats import pearsonr
import pandas as pd



casePath = 'normalizedCase.csv'
ctrlPath = 'normalizedCtrl.csv'

df = toDF(casePath)

def sort_frame_by_cluster(df, colname):
        return df.sort_values(colname)

def single_cluster_similarity(cluster):
    # distances = euclidean_distances(cluster, cluster)
    # distances = np.dot(cluster, cluster.T)
    distances = []
    # print type(cluster)
    # print cluster
    # in_clus_test = np.mean(cosine_similarity(cluster))
    # print 'incluster', np.mean(euclidean_distances(cluster))
    for index, clusterx in cluster.iterrows():
        # print clusterx[:-2]
        for index2, cluster2 in cluster.iterrows():
            if index != index2:
                 distances.append(np.dot(clusterx[:-2], cluster2.T[:-2]))

    # distances = cosine_similarity(cluster, cluster)
    # print cluster.shape
    # print distances
    return np.mean(distances)

def in_cluster_similarity(df, colname):
    sorted_frame = sort_frame_by_cluster(df, colname)
    set_cluster = set(list(sorted_frame[colname]))
    similarity_list = []
    cos_sim_list = []
    for cluster in set_cluster:
        subset = sorted_frame.loc[sorted_frame[colname] == cluster]
        av_dist = single_cluster_similarity(subset)
        similarity_list.append(av_dist)
        # cos_sim_list.append(cos_sim)
    # print 'in cluster', np.mean(cos_sim_list)
    return np.mean(similarity_list)

def between_cluster_similarity(clusters):
    # return np.mean(euclidean_distances(clusters))
    # for cluster in clusters:
    #     print cluster
    # return np.mean(np.dot(clusters))
    # print 'clusters', clusters
    # cluster1 = clusters.loc[1]
    # print cluster1
    # print 'norm', np.linalg.norm(cluster1)
    # print np.dot(cluster1, cluster1.T)
    # distance_mat = np.dot(clusters, clusters.T)
    # print 'between cluster: ', np.mean(euclidean_distances(clusters))
    distances = []
    for index, cluster in clusters.iterrows():
        for index2, cluster2 in clusters.iterrows():
            if index != index2:
                distances.append(np.dot(cluster[:-2], cluster2.T[:-2]))
    return np.mean(distances)

results = pd.DataFrame(index=np.arange(2,101), columns=['S_euc', 'D_euc', 'S/D_euc', 'S_cos', 'D_cos', 'S/D_cos'])

def pearson_coorelation_as_distance_measure(X):
    return pearsonr(X)

def dot_product(X):
    return np.dot(X, X.T)
sil = []

for i in range(2, 101):
    print 'on k = ', i
    kmeanseuc = KMeans(n_clusters=i)

    kmeanseuc.fit(df)
    silhouette_avg = silhouette_score(np.array(df), kmeanseuc.labels_)
    print 'silhouette score: ', silhouette_avg
    sil.append(silhouette_avg)
    print 'elbow method: ', sum(np.min(cdist(df, kmeanseuc.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0]
    # df_a = pd.DataFrame(kmeanseuc.labels_)
    # KMeans.euclidean_distances = dot_product
    # kmeanscos = KMeans(n_clusters=i)

    # df_new = kmeanscos.fit(df).transform(df)
    # df_new.fit()
    # print kmeanscos.transform(df, kmeanscos.cluster_centers_)
    # labelscos = kmeanscos.labels_

    # sorted_centers = np.sort(kmeanscos.cluster_centers_, axis=0)

    # df['labelseuc'] = kmeanseuc.labels_
    # df['labelscos'] = kmeanscos.labels_
    # df_centroids_euc = pd.DataFrame(kmeanseuc.cluster_centers_)
    # df_centroids_cos = pd.DataFrame(kmeanscos.cluster_centers_)
    # S_euc = in_cluster_similarity(df, 'labelseuc')
    # S_cos = in_cluster_similarity(df, 'labelscos')
    # print df_centroids_cos.shape
    # print 'Euclidean: ', np.mean(euclidean_distances(df_centroids_cos))
    # print 'Cosine: ', np.mean(cosine_similarity(df_centroids_cos))
    # print 'Dot Product: ', np.mean(dot_product(df_centroids_cos))
    # D_euc = between_cluster_similarity(df_centroids_euc)
    # D_cos = between_cluster_similarity(df_centroids_cos)
    # SD_euc = S_euc/D_euc
    # SD_cos = S_cos/D_cos
    # results.set_value(i, 'S_euc', S_euc)
    # results.set_value(i, 'D_euc', D_euc)
    # results.set_value(i, 'S/D_euc', SD_euc)
    # results.set_value(i, 'S_cos', S_cos)
    # results.set_value(i, 'D_cos', D_cos)
    # results.set_value(i, 'S/D_cos', SD_cos)



print sil
# df_sim = pd.DataFrame.from_dict(sim_list)
# df_sim.to_csv('inCluster.csv')
# results.to_csv('resultsNumpyDotProductSimilarity.csv')
