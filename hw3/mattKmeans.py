import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from fancyimpute import KNN
from scipy.stats import pearsonr


'''
    Problem 1: impute missing values
'''
py.sign_in('Mranftle', 'MikQahEQpuPow1Dd5XxZ')

# read in csv, change missing values to NaN
all_patient_data = pd.read_csv("allinput.csv", )
classes = list(all_patient_data['Classes'])
all_patient_data.replace('?', 'NaN', inplace=True)
all_patient_data = all_patient_data.apply(pd.to_numeric, errors='coerce')

# get and save classes for look-up later
class_lookup = list(all_patient_data['Classes'])

# group by class
ctrl_data = all_patient_data[all_patient_data['Classes'] == 1]
patient_data = all_patient_data[all_patient_data['Classes'] == 2]

# drop Classes
del ctrl_data['Classes']
del patient_data['Classes']

# save feature and sample names
feature_genes = ctrl_data.columns

# impute missing values using KNN, put back into a data frame
ctrl_impute = KNN(k=7).complete(ctrl_data)
patient_impute = KNN(k=7).complete(patient_data)
ctrl_impute_df = pd.DataFrame(ctrl_impute, columns=feature_genes)
patient_impute_df = pd.DataFrame(patient_impute, columns=feature_genes)

# all imputed data
all_imputed_data = pd.concat([patient_impute_df, ctrl_impute_df], ignore_index=True)

def get_indexes(labels, value):
    return [x for x in range(len(labels)) if labels[x] == value]
'''
    Problem 2: bottom-up clustering
'''
# # custom affinity to use pearson correlation
def pearson_affinity(M):
   return 1 - np.array([[pearsonr(a,b)[0] for a in M] for b in M])

# # level 1
agg_cluster = AgglomerativeClustering(n_clusters=2, affinity=pearson_affinity, linkage="complete").fit(all_imputed_data)
print 'agg_1', get_indexes(agg_cluster.labels_, 0)
print 'agg_2', get_indexes(agg_cluster.labels_, 1)

# get data points that were wrongly classified
agg_wrong_labels = []
for c in range(len(class_lookup)):
    if float(agg_cluster.labels_[c]) != float(class_lookup[c])-1:
        agg_wrong_labels.append(c)

print 'bottom-up \n'
print 'wrong: ' + str(len(agg_wrong_labels)) + '/' + str(len(class_lookup))
print 1-(float(len(agg_wrong_labels)) / float(len(class_lookup)))

print 'wrongly classified: \n' + str(agg_wrong_labels)
print " "

# level 2, recluster each group
agg_cluster2 = AgglomerativeClustering(n_clusters=4, affinity=pearson_affinity, linkage="complete").fit(all_imputed_data)

# plot pie chart
agg_cluster2_values = Counter(agg_cluster2.labels_).values()
agg_cluster2_values = [float(x)*100/sum(agg_cluster2_values) for x in agg_cluster2_values]

clust_fig_2 = {
    'data': [{'values': agg_cluster2_values,
              'type': 'pie'}],
    'layout': {'title': 'Level 2 Bottom Up Clustering Label Distribution'}
     }

py.iplot(clust_fig_2, filename="clust_fig_2")

# level 3
agg_cluster3 = AgglomerativeClustering(n_clusters=8, affinity=pearson_affinity, linkage="complete").fit(all_imputed_data)
agg_cluster3_values = [0,0,0,0,0,0,0,0]
for i in agg_cluster3.labels_:
    agg_cluster3_values[i] += 1
agg_cluster3_values = [float(x)*100/sum(agg_cluster3_values) for x in agg_cluster3_values]

clust_fig_3 = {
    'data': [{'values': agg_cluster3_values,
              'type': 'pie'}],
    'layout': {'title': 'Level 3 Bottom Up Clustering Label Distribution'}
     }

py.iplot(clust_fig_3, filename="clust_fig_3")

'''
    Problem 3: top-down clustering
'''
# use pearson coorelation as distance measure
def pearson_coorelation_as_distance_measure(X):
    return pearsonr(X)

KMeans.euclidean_distances = pearson_coorelation_as_distance_measure

kmeans_cluster = K
KMeans(n_clusters=2).fit(all_imputed_data)

# get data points that were wrongly classified
kmeans_wrong_labels = []
for c in range(len(class_lookup)):
    if kmeans_cluster.labels_[c] != (class_lookup[c]-1):
        kmeans_wrong_labels.append(c)

print 'top-down \n'
print 'wrong: ' + str(len(kmeans_wrong_labels)) + '/' + str(len(class_lookup))
print 1 - float(len(kmeans_wrong_labels)) / float(len(class_lookup))

print 'wrongly classified: \n' + str(kmeans_wrong_labels)
print " "
print 'kmeans_1', get_indexes(kmeans_cluster.labels_, 0)
print 'kmeans_2', get_indexes(kmeans_cluster.labels_, 1)

print 'data_labels_1', get_indexes(class_lookup, 1)
print 'data_labels_2', get_indexes(class_lookup, 2)



# run recursively
kmeans_cluster_data2_ctrl = all_imputed_data.iloc[[x for x in range(len(kmeans_cluster.labels_)) if kmeans_cluster.labels_[x] == 0]]
kmeans_cluster_data2_patient = all_imputed_data.iloc[[x for x in range(len(kmeans_cluster.labels_)) if kmeans_cluster.labels_[x] == 1]]
cluster_data2 = [kmeans_cluster_data2_ctrl, kmeans_cluster_data2_patient]

# run clustering recursively
def kmeans_levels(data, level):
    if level > 3:
        return

    # cluster each dataframe passed
    all_labels = []
    next_data = []
    for d in data:
        temp_cluster = KMeans(n_clusters=2).fit(d)
        indices = d.index
        next_data.append(all_imputed_data.loc[[x for x in range(len(temp_cluster.labels_)) if temp_cluster.labels_[x] == 0]])
        next_data.append(all_imputed_data.loc[[x for x in range(len(temp_cluster.labels_)) if temp_cluster.labels_[x] == 1]])
        all_labels.append(list(temp_cluster.labels_))

    # combine labels from all groups
    i = 0
    combined_labels = []
    for g in all_labels:
        for l in g:
            combined_labels.append(l+i)
        i = i + 2

    # plot percentages
    temp_values = Counter(combined_labels).values()
    temp_values = [float(x)*100/sum(temp_values) for x in temp_values]

    title = 'Level ' + str(level) + ' Top Down Clustering Label Distribution'
    filename = "kmeans_clust_level_" + str(level)
    clust_fig_2 = {
        'data': [{'values': temp_values,
                  'type': 'pie'}],
        'layout': {'title': title}
         }

    py.plot(clust_fig_2, filename=filename)

    # recurse
    kmeans_levels(next_data, level + 1)

kmeans_levels(cluster_data2, 2)

'''
    Part 2: Problem 1
'''
# get eigen vectors and eigen values
all_patient_pca = PCA()
all_patient_pca.fit_transform(all_imputed_data)
top_eigens = sorted(zip(all_patient_pca.explained_variance_, all_patient_pca.components_), key=lambda x:x[0], reverse=True)

# make plot of eigen values using plotly api
trace = go.Scatter(x=range(len(all_patient_pca.explained_variance_)), y=all_patient_pca.explained_variance_)
graph_points =[trace]
py.sign_in('Mranftle', 'MikQahEQpuPow1Dd5XxZ')
py.iplot(graph_points, filename='eigenvalues')

'''
    Part 2: Problem 2
'''
# run kmeans using top k eigen vectors as features
k = 2
correct_classify = []
while k <= 20:
    feature_vec = pd.DataFrame([x[1] for x in top_eigens[:k]]).T
    transformed_data = pd.DataFrame(feature_vec.values.T.dot(all_imputed_data.values.T)).T


    kmeans_pca = KMeans(n_clusters=2).fit(transformed_data)
    wrong = 0.0
    for c in range(len(class_lookup)
                   ):
        if kmeans_pca.labels_[c] != (class_lookup[c]-1):
            wrong = wrong + 1
    print k, ': ', wrong/float(len(class_lookup))
    correct_classify.append(wrong/float(len(class_lookup)))
    k = k + 1

# make plot using plotly api
trace = go.Scatter(x=range(2, 20), y=correct_classify)
graph_points =[trace]
py.sign_in('Mranftle', 'MikQahEQpuPow1Dd5XxZ')
py.iplot(graph_points, filename='kmeans_pca')


'''
    Part 3: Bonus
'''
samples, features = all_imputed_data.shape

k = 2
n = samples

U, s, V = np.linalg.svd(all_imputed_data, full_matrices=False)

svd = TruncatedSVD(n_components=n)

svd.fit(all_imputed_data)

eigvals = svd.explained_variance_
eigvecs = svd.components_

k_eigs = sorted(zip(eigvals, eigvecs), key=lambda x:x[0], reverse=True)

trace3 = go.Scatter(x=np.arange(len(eigvals)), y=eigvals)
data3 = [trace3]


py.iplot(data3, filename='SVD_')
k = 2

incorrect_svd = []
while k<=20:
    sig = np.mat(np.eye(k)*s[:k])
    features = pd.DataFrame([x[1] for x in k_eigs[:k]]).T
    trans_svd = pd.DataFrame(features.values.T.dot(all_imputed_data.values.T)).T
    new = U[:,:k]*sig*V[:k,:]
    svd_kmeans = KMeans(n_clusters=2).fit(new)
    wrong = 0.0
    for i in range(len(class_lookup)):
        if svd_kmeans.labels_[i] != (class_lookup[i] - 1):
            wrong += 1
    pctw = wrong/float(len(class_lookup))
    #print(k, ':', pctw)
    incorrect_svd.append(pctw)
    k += 1

print(incorrect_svd)
trace4 = go.Scatter(x = np.arange(2, 20), y=incorrect_svd)
data4 = [trace4]
py.iplot(data4, filename='svd-kmeans')
