from formatData import processData
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


dataPath = '/Users/macuser/Downloads/514hw1csv.csv'

train_data, test_data = processData(dataPath)

features = list(train_data.columns[2:])


X = train_data[features]
y = train_data['TIME']

X_test = test_data[features]
y_real = test_data['TIME']

LogReg = LogisticRegression()
LogReg.fit(X, y)

log_predict = LogReg.predict(X_test)
logrecacc = accuracy_score(y_real, log_predict)

print "Logistic Regression"
print "Predicted: ", list(log_predict)
print "Accuracy Score: ", logrecacc
print "Actual Values: ", list(y_real)
print

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)
n1 = neigh.predict(X_test)
n1acc = accuracy_score(y_real, n1)

print "kNN n=1"
print "Predicted: ", list(n1)
print "Accuracy Score: ", n1acc
print "Actual Values: ", list(y_real)
print

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
n3 = neigh.predict(X_test)
n3acc = accuracy_score(y_real, n3)

print "kNN n=3"
print "Predicted: ", list(n3)
print "Accuracy Score: ", n3acc
print "Actual Values: ", list(y_real)
print

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, y)
n5 = neigh.predict(X_test)
n5acc = accuracy_score(y_real, n5)

print "kNN n=5"
print "Predicted: ", list(n5)
print "Accuracy Score:", n5acc
print "Actual Values: ", list(y_real)
print

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X, y)
dtpredictions = dt.predict(X_test)
dtacc = accuracy_score(y_real, dtpredictions)
print "Decision Tree"
print "Predicted: ", list(dtpredictions)
print "Accuracy Score:", dtacc
print "Actual Values: ", list(y_real)
print


dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
dt.fit(X, y)
prunepredict = dt.predict(X_test)
pruneacc = accuracy_score(y_real, prunepredict)
print "Decision Tree with pruning"
print "Predicted: ", list(prunepredict)
print "Accuracy Score:", pruneacc
print "Actual Values: ", list(y_real)
print


rf = RandomForestClassifier(n_estimators=20, criterion='entropy', bootstrap=True)
rf.fit(X, y)
rfpredict = rf.predict(X_test)
rfacc = accuracy_score(y_real, rfpredict)
print "Random Forest"
print "Predicted: ", list(rfpredict)
print "Accuracy Score:", rfacc
print "Actual Values: ", list(y_real)
print


svm = SVC(kernel="linear")
svm.fit(X, y)
svmpredict = svm.predict(X_test)
svmacc = accuracy_score(y_real, svmpredict)
print "Support Vector Machine"
print "Predicted: ", list(svmpredict)
print "Accuracy Score:", svmacc
print "Actual Values: ", list(y_real)
print











