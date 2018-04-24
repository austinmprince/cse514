import pandas as pd
from math import isnan
from scipy.stats import norm
import numpy as np
import sys

ctrlPath = 'ctrlCopy.csv'
casePath = 'caseCopy.csv'



def convertframe(dataPath, frac=1):
    # read in the data and get only first 1000 rows
    data = pd.read_csv(dataPath)
    # transpose so patients are the rows and columns are the genes
    data = data.transpose()
    col_names = list(data.iloc[0])
    # Create a dictionary that matches integer values to the genes and replace the genes as the column names
    # for our data frame
    rename_dict = {}
    for colname in data.columns:
        rename_dict[colname] = col_names[colname]

    data.rename(columns=rename_dict, inplace=True)
    data = data.iloc[1:]
    return data.sample(frac=frac)


def getGaussDistribution(dataPath,  frac=1):
    dataFrame = convertframe(dataPath, frac=frac)
    for column in dataFrame:
        # change classes to a boolean 0 or 1 from values of 1.0 or 2.0
        if column == 'Classes':
            dataFrame.loc[(dataFrame.Classes == 1.0, 'Classes')] = 0
            dataFrame.loc[(dataFrame.Classes == 2.0, 'Classes')] = 1
            for index, value in enumerate(dataFrame[column].values):
                if value == '1':
                    dataFrame[column][index] = 1

        else:
            # make a list of existing values
            dataList = list(dataFrame[column].values)
            gaussDist = []
            for index, value in enumerate(dataList):
                if type(value) is not float:
                    try:
                        # convert non floats to a float ie 1E-5
                        dataList[index] = float(value)
                        value = float(value)
                        dataFrame[column][index] = float(value)
                    except:
                        # if not a float and can't be converted convert to NaN
                        dataList[index] = float('NaN')
                        value = float('NaN')
                # if value is a number put in list to get gaussian deistribution
                if not isnan(value):
                    gaussDist.append(value)
            # check to make sure this is stddev not var
            mu, std = norm.fit(gaussDist)
            # generate random number in Gaussian distribution for unknown values
            for index, value in enumerate(dataFrame[column]):
                if isnan(dataList[index]):
                    rnum = np.random.normal(mu, std)
                    dataList[index] = rnum
                    dataFrame[column][index] = rnum
    return dataFrame

def topPercent(dataPath, percent, fileName):
    dataFrame = getGaussDistribution(dataPath)
    dataFrame = dataFrame.sort_index(ascending=False)
    patients = list(dataFrame.index)
    numPatients = int(len(patients) * percent)
    for gene in dataFrame:
        if gene == 'Classes':
            continue
        sortedDataFrame = dataFrame.sort_values(gene, ascending=False)
        topPatients = sortedDataFrame.head(numPatients)[gene]
        topPatients = set(topPatients.index)

        for i in range(len(dataFrame[gene])):
            patient = patients[i]
            if patient in topPatients:
                dataFrame[gene][patient] = 1
            else:
                dataFrame[gene][patient] = 0
    dataFrame.to_csv(fileName)
    return dataFrame



topPercent(ctrlPath, 0.1, 'top10')