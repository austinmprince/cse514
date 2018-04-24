import formatData as fd
import math
import pandas as pd
from util import PriorityQueue


dataPath = '/Users/macuser/Downloads/514hw1csv.csv'
#
# def euclideanDistance(point1, point2, pqueue):
#     print point
#      # trainValues = list.loc[:, (list.columns != 'DATE') & (list.columns != 'TIME')]
#      for index, row in list.iterrows():
#         tempdist = pow(row['TEMP'] - point['TEMP'], 2)
#         humdist = pow(row['HUM'] - point['HUM'], 2)
#         lightdist = pow(row['LIGHT'] - point['LIGHT'], 2)
#         clouddist = pow(row['CLOUD'] - point['CLOUD'], 2)
#         totaldist = math.sqrt(tempdist + humdist + lightdist + clouddist)
#         return totaldist

def euclideanDistance(point1, point2):
    tempdist = pow(point1['TEMP'] - point2['TEMP'], 2)
    humdist = pow(point1['HUM'] - point2['HUM'], 2)
    lightdist = pow(point1['LIGHT'] - point2['LIGHT'], 2)
    clouddist = pow(point1['CLOUD'] - point2['CLOUD'], 2)
    totaldist = math.sqrt(tempdist + humdist + lightdist + clouddist)
    return totaldist


def getNeighbors(testInstance, trainingSet, k):
    pqueue = PriorityQueue()
    nearestNeighbor = {}
    for key, value in trainingSet.iterrows():
        dist = euclideanDistance(value, testInstance)
        pqueue.push(key, dist)
    for i in range(k):
        dist,key = pqueue.pop()
        nearestNeighbor[key] = dist
    return nearestNeighbor



def getPrediction(neighbors, trainFrame):
    predict = 0
    for index, dist in neighbors.iteritems():
        # print index, dist
        # print "%s, %s" %(index, dist)
        #
        # print trainFrame.loc[index]['TIME']
        predict += trainFrame.loc[index]['TIME']

    predict = int(round(predict/len(neighbors)))
    return predict


def getAccuracy(predictions, testData):
    correct = 0
    for key, value in predictions.iteritems():
        # print predictions[key][0]

        if predictions[key][0] == testData.iloc[key]['TIME']:
            # print "%s: %s" %(predictions[key], testData.iloc[key]['TIME'])
            # print "in here"
            correct += 1
    # print correct
    # print len(predictions)
    # print float(correct)/float(len(predictions))
    correct = float(correct)/float(len(predictions))
    # print correct
    return correct

def kNN(k, testData, trainData):
    predictions = {}
    for row, item in testData.iterrows():
        x = getNeighbors(item, trainData, k)
        y = getPrediction(x, trainFrame)
        predictions[row] = (y, x)
    return predictions


train_data, test_data = fd.processData(dataPath)
trainFrame = pd.DataFrame(train_data)
testFrame = pd.DataFrame(test_data)

# print trainFrame
#
# print testFrame
# print trainFrame


predictions = kNN(1, testFrame, trainFrame)

percentcorrect = getAccuracy(predictions, testFrame)

print percentcorrect








# trainValues = testFrame.loc[:, (testFrame.columns != 'DATE') & (testFrame.columns != 'TIME')]




















