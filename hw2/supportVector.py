from sklearn import svm
import numpy as np
from gaussianClassification import getGaussDistribution
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import csv

ctrlPath = 'ctrlCopy.csv'
copyPath = 'caseCopy.csv'

def supportVector(ctrlPath, copyPath):
    dataFrame = getGaussDistribution(ctrlPath, copyPath)
    #
    # df2['Classes'] = dataFrame['Classes']
    # y = list(df2['Classes'].values)
    features = list(dataFrame.columns[1:])
    # X = df2[features]
    X = dataFrame[features]
    X_norm = pd.DataFrame(preprocessing.normalize(X))
    y = list(dataFrame['Classes'].values)

    clf = svm.SVC(kernel='linear')
    # print X
    # print y
    # clf = svm.LinearSVC()
    clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(kernel='poly', degree=2, coef0=1))
    clef = make_pipeline(preprocessing.StandardScaler(), svm.SVC(kernel='poly', degree=2))
    # clf = svm.SVC(kernel='poly', degree=2, coef0=1)
    # clef = svm.SVC(kernel='poly', degree=2)
    # clef.fit(X_norm, y)
    # clf.fit(X_norm, y)
    inhomoscores = cross_val_score(clf, X, y, cv=10)
    homoscores = cross_val_score(clef, X, y, cv=10)
    # homoscores = cross_val_score(clef, X_norm, y, cv=10)
    print "inhomogeneous: ", np.mean(inhomoscores)
    print "homogeneous: ", np.mean(homoscores)
    return np.mean(homoscores), np.mean(inhomoscores)
    # print clf.coef_
    # return clf.coef_, features

# weights, features = supportVector(ctrlPath, copyPath)
# print weights
# print len(weights)

# for value in weights:
#     weights = value

# print weights
# print len(weights)
# coef = weights.ravel()
# top_positive_coefficients = np.argsort(coef)[-100:]
# print top_positive_coefficients

# indicies = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:100]
# topOne = {}
# topFeatures = []
# for index in indicies:
#      topOne[features[index]] = weights[index]
#      topFeatures.append(features[index])
# #
# #
# print topFeatures
# with open('linearSVM.csv', 'wb') as f:  # Just use 'w' mode in 3.x
#     w = csv.DictWriter(f, topOne.keys())
#     w.writeheader()
#     w.writerow(topOne)

homolist = []
inhomolist = []
for i in range(0, 10):
    homo, inhomo = supportVector(ctrlPath, copyPath)
    homolist.append(homo)
    inhomolist.append(inhomo)
print "homo mean: ", np.mean(homolist)
print "inhomo mean: ", np.mean(inhomolist)



