from sklearn import decomposition
import pandas as pd
import numpy as np
from preprocessing import toDF
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import sys

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

# mine

# casePath = 'normalizedCase.csv'
# ctrlPath = 'normalizedCtrl.csv'

# zelasO
casePath = 'caseNormalized.csv'
ctrlPath = 'controlNormalized.csv'

case_df = toDF(casePath)
ctrl_df = toDF(ctrlPath)

# print case_df.shape
# print ctrl_df.shape

#
df = pd.concat([ctrl_df, case_df])
# df = toDF('total_df_normalized.csv')


# del df['Classes']

y = [0] * 188
yz = [1] * 176

y_final = yz + y

pca = decomposition.PCA(n_components=3)
x = pca.fit_transform(df)

svm = SVC()
scores = cross_val_score(svm, x, y_final, cv=10)
print np.mean(scores)

#
# scores = cross_val_score(pca, y_final, cv=5)


plt.plot(x[0:188,0], x[0:188,1], 'o', markersize=7, color='blue', alpha=0.5, label='Control')
plt.plot(x[189:,0], x[189:,1], '^', markersize=7, color='red', alpha=0.5, label='Case')
plt.xlim([-50,50])
plt.ylim([-25,25])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')
plt.show()



# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# pca_3d = decomposition.PCA(n_components=3)
# x_3d = pca_3d.fit_transform(df)
#
#
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')
# plt.rcParams['legend.fontsize'] = 10
# ax.plot(x_3d[0:189,0], x_3d[0:189,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
# ax.plot(x_3d[188:,0], x_3d[188:,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
#
# plt.title('Samples for class 1 and class 2')
# ax.legend(loc='upper right')
#
# plt.show()
# k = 3
# pca = decomposition.PCA(n_components = k)
# transform = pca.fit_transform(df)
#
#
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')
# plt.rcParams['legend.fontsize'] = 10
#
# control_class = transform[0:188]
# case_class = transform[188:]
# ax.plot(control_class[:,0], control_class[:,1], control_class[:,2], 'o', markersize=8, color='green', alpha=0.5, label='Control')
# ax.plot(case_class[:,0], case_class[:,1], case_class[:,2], '^', markersize=8, alpha=0.5, color='blue', label='AD')
# ax.set_xlim3d(-40, 60)
# ax.set_ylim3d(-20,40)
# ax.set_zlim3d(-20,20)
# plt.title('Samples for class 1 and class 2')
# ax.legend(loc='upper right')
#
# plt.show()

# plt.show()




# top_eigenvalues = sorted(zip(pca.explained_variance_, pca.components_), key=lambda x:x[0], reverse=True)
# information_ratio = sorted(pca.explained_variance_ratio_, reverse=True)
# print len(information_ratio)
# sum_ratio = []
# for i in range(len(information_ratio)):
#     sum_ratio.append(sum(information_ratio[0:i]))
# sum_ratio = sum_ratio[1:]
# print sum_ratio
# for i in range(len(sum_ratio)):
#     if sum_ratio[i] - 0.8 < 0.01:
#         print i
#         print sum_ratio[i]
# markers_on = [12]
# plt.plot(sum_ratio, markevery=markers_on)
# plt.xlabel('Number of Components')
# plt.ylabel('Accumulative Information')
# plt.title('Accumulative Information vs Num Components')
# plt.show()

# plt.plot(pca.explained_variance_)
# plt.xlabel('Number of Components')
# plt.ylabel('Eigenvalues')
# plt.title('Eigenvalues vs Num Components')
# plt.show()



