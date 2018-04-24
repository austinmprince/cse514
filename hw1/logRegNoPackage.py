import pandas as pd
import formatData as fd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

dataPath = '/Users/macuser/Downloads/514hw1csv.csv'


train_data, test_data = fd.processData(dataPath)

trainFrame = pd.DataFrame(train_data)
testFrame = pd.DataFrame(test_data)



X_train = trainFrame.ix[:,(1, 2, 3, 4)].values
y_train = trainFrame.ix[:,0].values



X_test = testFrame.ix[:,(1, 2, 3, 4)].values
y_test = testFrame.ix[:,0].values


LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

y_predict = LogReg.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_predict)
print confusion_matrix







# logit = sm.Logit(train_data['TIME'], train_data[train_cols])
#
# # fit the model
# result = logit.fit()

# print result.summary()




