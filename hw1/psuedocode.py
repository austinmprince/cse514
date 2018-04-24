for attribute a:
    for all splits in attribute a:
        find infogain of split on attribute a with split y1<x<y2 # for continuous variable with values from y1 to y2
    split on attribute a at split x that gives max infogain
    create a nodes from the split x with attribute a
repeat until all samples split and classified into singular classifications # every sample in train_set has been classified


for i in range(0: num_trees):
    select a random sample containing .8 percent of train_Set
    train a decision tree based on this sample
    append this tree to tree_list
for datapoint in test_data:
    run datapoint through tree_list to classify
    append classifications return from every tree in tree_list to predict list
    take mode of classifications to predicut datapoint


def kNN(k, test_set, train_set):
    for datapoint in test_set:
        find train_points in train_set that are closest to datapoint
        classify train_points
        clasify datapoint based on majority vote of train_points
        if train_points is tied:
            classify datapoint randomly
    return predictions for test_set
