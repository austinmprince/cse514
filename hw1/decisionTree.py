from formatData import processData
import subprocess
# import pandas
# import math
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

dataPath = '/Users/macuser/Downloads/514hw1csv.csv'

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
       subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

train_data, test_data = processData(dataPath)



features = list(train_data.columns[2:])
# for v in train_data[(train_data['TEMP'] > 9) & (train_data['LIGHT'] < 10.1)]['TIME'].values:
#     print(v)
# sortedcolumns = train_data.sort(['TEMP', 'LIGHT', 'HUM'])
# print sortedcolumns



y = train_data['TIME']
X = train_data[features]

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X, y)

test_features = list(test_data.columns[2:])
test_set = test_data[features]
test_1 = test_data

print test_set
predictions = dt.predict(test_data[features])

print predictions

visualize_tree(dt, features)

# dot_data = tree.export_graphviz(dt, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("geese")





# class Node(object):
#     def __init__(self, data):
#         self.data = data
#         self.children = []
#
#     def add_child(self, obj):
#         self.children.append(obj)
#

#
#
# for column in train_data:
#     if (column == 'DATE') or (column == 'TIME'):
#         pass
#     else:
#         sortedcolumns = train_data.sort_values(by=[column])
#         for index, value in sortedcolumns.iterrows():
#             if index == sortedcolumns.ix[0]:
#                 break













