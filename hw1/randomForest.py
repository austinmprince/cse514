from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from formatData import processData
from sklearn.metrics import accuracy_score

dataPath = '/Users/macuser/Downloads/514hw1csv.csv'

train_data, test_data = processData(dataPath)
# print train_data

features = list(train_data.columns[2:])
# print features
# print train_data
# print test_data

y = train_data['TIME']
X = train_data[features]
print type(y)
print X

rf = RandomForestClassifier(n_estimators=20, criterion='entropy', bootstrap=True)

rf.fit(X, y)
predicitions = rf.predict(test_data[features])

print predicitions
print accuracy_score(test_data['TIME'], predicitions)


