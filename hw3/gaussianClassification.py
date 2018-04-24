import numpy as np
from scipy.stats import norm
from fileReading import convertframe, concatenateFrames
from math import isnan
import sys


ctrlPath = 'ctrl.csv'
copyPath = 'case.csv'

def getGaussDistribution(path, frac=1):
    # dataFrame = concatenateFrames(ctrlPath, copyPath, frac=frac)
    dataFrame = convertframe(path)
    # print dataFrame
    for column in dataFrame:
        if column == 'Classes':
            dataFrame.loc[(dataFrame.Classes == 1.0, 'Classes')] = 0
            dataFrame.loc[(dataFrame.Classes == 2.0, 'Classes')] = 1
            for index, value in enumerate(dataFrame[column].values):
                if value == '1':
                    dataFrame[column][index] = 1
                    # print value
                    # print index
                    # print dataFrame[column][index]
        # if column != 'Classes':
           # print dataFrame[column].values
        else:
            dataList = list(dataFrame[column].values)
            # print dataFrame[column].values

            gaussDist = []
            # print dataList
            for index, value in enumerate(dataList):
                if type(value) is not float:
                    try:
                        dataList[index] = float(value)
                        value = float(value)
                        dataFrame[column][index] = float(value)
                    except:
                        dataList[index] = float('NaN')
                        value = float('NaN')
                if not isnan(value):
                    gaussDist.append(value)
            # check to make sure this is stddev not var
            mu, std = norm.fit(gaussDist)
            # generate random number in Gaussian distribution for unknown values
            for index, value in enumerate(dataFrame[column]):
                # dataFrame.set_value()

                if isnan(dataList[index]):
                    rnum = np.random.normal(mu, std)
                    dataList[index] = rnum
                    dataFrame[column][index] = rnum
    return dataFrame





# results = getGaussDistribution(ctrlPath)
# print results
# gaussDistribution(ctrlData)
# print ctrlData['GI_10800414-S']['WGACON84']
