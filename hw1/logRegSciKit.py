import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import csv
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report







dataPath = '/Users/macuser/Downloads/514hw1csv.csv'

allData = pd.read_csv(dataPath)


allData.loc[(allData.TINE > 0, 'TINE')] = 1
allData.loc[(allData.TINE <= 0, 'TINE')] = 0

allData.dropna(how='any', inplace=True)
allData.to_csv("editedHW1.csv", index=False)


# trainData = pd.read_csv(dataPath, nrows=5)
# testData = pd.read_csv(dataPath, skiprows=5)

# df.loc[(df.Name.isin(['Avi','Dav','Ron'])) & (df.Age < 33), 'Babys'] += 1

# trainData.loc[(trainData.TINE > 0, 'TINE')] = 1
# trainData.loc[(trainData.TINE <= 0, 'TINE')] = 0
# testData.loc[(trainData.TINE > 0, 'TINE')] = 1
# testData.loc[(trainData.TINE <= 0, 'TINE')] = 0



# print trainData
#
# # print trainData.head()
# print trainData['TINE']
#
# for data in trainData['TINE']:
#     if data > 0:
#         print 'big boi'
#     else:
#         print 'little boi'
#


# def convertToPlusMinus(data):
#     for point in data:
#         if


