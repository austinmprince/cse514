from gaussianClassification import getGaussDistribution
from sklearn import preprocessing
import pandas as pd
from fileReading import convertframe
import numpy as np

import sys

ctrlPath = 'ctrl.csv'
casePath = 'case.csv'
imputedCase = 'imputedCase.csv'
imputedCtrl = 'imputedCtrl.csv'


def toDF(path):
    data = pd.read_csv(path, index_col=0)
    return data
    # col_names = list(data.loc[0])


def normalize(path):
    df = toDF(path)
    # df.drop(labels='Classes', axis=1, inplace=True)
    std_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(df.transpose())
    df_std = std_scale.transform(df.transpose())
    data = pd.DataFrame(np.transpose(df_std), index=df.index, columns=df.columns)
    # data.to_csv('normalizedCase.csv')
    return data


# writes the imputed csv with a gaussian distribution to a csv
# so that I don't have to keep calling it
def writeGaussian(path):
    dataFrame = getGaussDistribution(path)
    dataFrame.to_csv('imputedCase.csv', columns=list(dataFrame))


#
# writeGaussian(ctrlPath)

case_df = toDF(imputedCase)
ctrl_df = toDF(imputedCtrl)
total_df = pd.concat([ctrl_df, case_df])
total_df = total_df.to_csv('total_df.csv')


normalized_all = normalize('total_df.csv')
normalized_all.to_csv('total_df_normalized.csv')

# normalizedCtrl = normalize(imputedCase)
# normalizedCtrl.to_csv('normalizedCase.csv', columns=list(normalizedCtrl))
