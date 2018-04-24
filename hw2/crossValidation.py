from sklearn.ensemble import RandomForestClassifier
from gaussianClassification import getGaussDistribution

from sklearn.cross_validation import cross_val_score

import numpy as np

ctrlPath = 'ctrlCopy.csv'
copyPath = 'caseCopy.csv'

dataFrame = getGaussDistribution(ctrlPath, copyPath, frac=0.55)
y = list(dataFrame['Classes'].values)
print "55 percent instances, 70 percent features"

features = list(dataFrame.columns[1:])
X = dataFrame[features]

range = np.arange(0.5, 0.95, 0.05)
rf = RandomForestClassifier(n_estimators=500, criterion='entropy', bootstrap=True, max_features=0.7)

print "made tree"
scores = cross_val_score(rf, X, y,)
print scores
print np.mean(scores)

