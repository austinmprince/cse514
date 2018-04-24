from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from gaussianClassification import getGaussDistribution
import numpy as np
import pandas as pd


ctrlPath = 'ctrlCopy.csv'
copyPath = 'caseCopy.csv'

def randomForest(featureFrac, instanceFrac, ctrlPath, copyPath, feat=True):
    dataFrame = getGaussDistribution(c, copyPath, frac=instanceFrac)

    y = list(dataFrame['Classes'].values)
    print dataFrame['Classes'].values
    features = list(dataFrame.columns[1:])
    X = dataFrame[features]
    # print "percent features", featureFrac

    rf = RandomForestClassifier(n_estimators=500, criterion='entropy', bootstrap=True, max_features=featureFrac)
    print X
    print y
    scores = cross_val_score(rf, X, y, cv=10)

    print np.mean(scores)
    # print 'featureFrac: %s, instanceFrac: %s' %(featureFrac, instanceFrac)
    # print scores.mean()
    # rf.fit(X, y)

    importance = rf.feature_importances_
    estimators = rf.estimators_
    # print estimators
    return estimators, features
    # print len(importance)
    # print type(importance)
    # array of the indices of the 30 most important genes in our RF
    # mostImport = sorted(range(len(importance)), key=lambda i: importance[i])[-30:]
    # create a dict to store the values and the genes that are important for this iteration
    # importDict = {}
    # for i in mostImport:
    #     print importance[i]
    #     print features[i]
    #     importDict[features[i]] = importance[i]
    # if feat == True:
    #     barPlot(importDict, featureFrac, True)
    # else:
    #     barPlot(importDict, instanceFrac, False)
    # return importDict
# estimators are the individual trees that are returned, features are the attributes
# that the tree is split on we need this for future use
# estimators, features = randomForest(0.7, 0.50, ctrlPath, copyPath, False)
#
#
# # print "num of trees", len(estimators)
# freqList = []
# for estimator in estimators:
#
#     # print "num features", tree.n_features_
#     # print "num classes", tree.n_classes_
#     # print "max features", tree.max_features_
#     # print "num outputs", tree.n_outputs_
#     # print estimator.get_params()
#     # n_nodes = estimator.tree_.node_count
#     # children_left = estimator.tree_.children_left
#     # children_right = estimator.tree_.children_right
#
#     feature = estimator.tree_.feature
#     # print "num of features split on", len(feature)
#     for feature in estimator.tree_.feature:
#         freqList.append(features[feature])
# freqDict = {x:freqList.count(x) for x in freqList}
# print freqDict
#     # sortedDict = sorted(freqDict.items(), key=operator.itemgetter(1))
#     # print sortedDict
# # estimatorOne = estimators[1]
# # feature = estimatorOne.tree_.feature
# #     for feature in estima
#     # threshold = estimator.tree_.threshold
#
#
#
#
#     # importances = tree.feature_importances_
#     # print "important features", importances
#     # print len(importances)
#     # print np.count_nonzero(importances)
# estimators, features = randomForest(0.7, 0.5, ctrlPath, copyPath)
# for estimator in estimators:
#     treefeatures = estimator.tree_.feature
#     print treefeatures


def forThreshold(ctrlPath, copyPath):
    range = np.arange(0.5, 0.95, 0.05)
    finalList = []
    for frac in range:
        estimators, features = randomForest(frac, 0.7, ctrlPath, copyPath, False)
        freqList = []
        for estimator in estimators:
            feature = estimator.tree_.feature
            for feature in estimator.tree_.feature:
            # print features[feature]
                freqList.append(features[feature])
        freqDict = {x:freqList.count(x) for x in freqList}
        # sortedList = sorted(freqDict.items(), key=operator.itemgetter(1,0), reverse=True)
        finalList.append(freqDict)


    frame = pd.DataFrame(finalList)
    frame.to_csv('frequencyInstance.csv')



def crossValidation(ctrlPath, copyPath):
    range = np.arange(0.5, 0.95, 0.05)
    finalList = []
    for frac in range:
        randomForest(frac, 0.7, ctrlPath, copyPath, False)
    return "done"

crossValidation(ctrlPath, copyPath)
# forThreshold(ctrlPath, copyPath)