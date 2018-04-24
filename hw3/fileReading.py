import pandas as pd


ctrlData = 'ctrl.csv'
copyData = 'case.csv'
ctrlInvert = 'ctrlinvert.csv'
copyInvert = 'copyinvert.csv'

# puts the csv data into a pandas data frame returns a random fraction of the data frame depending
# on the user specified value
def convertframe(dataPath, frac=1):
    # read in the data and get only first 1800 rows
    # col_names = pd.read_csv(inverse, nrows=1)
    # print col_names
    data = pd.read_csv(dataPath)
    # transpose so patients are the rows and columns are the genes
    data = data.transpose()
    # print data
    col_names = list(data.iloc[0])
    # print col_names

    # data.rename(col_names)
    # Create a dictionary that matches integer values to the genes and replace the genes as the column names
    # for our data frame
    rename_dict = {}
    for colname in data.columns:
        rename_dict[colname] = col_names[colname]
    # print rename_dict
    data.rename(columns=rename_dict, inplace=True)
    # print list(data.columns.values)
    # print data['GI_10047091-S']
    data = data.iloc[1:]
    # print data
    return data

def concatenateFrames(casePath, ctrlPath, frac=1):
    ctrlFrame = convertframe(ctrlData)
    copyFrame = convertframe(copyData)
    frames = [ctrlFrame, copyFrame]
    finalFrame = pd.concat(frames)
    return finalFrame.sample(frac=frac)




# print copyFrame
# s = data.ix[0,:]
# extracts a random sample of 70 percent of the data from the data frame
# data.sample(frac=0.7)
# print list(data.iloc[0])
# print copyFrame.ix['WGAAD1']
# print ctrlFrame.rows.values

df = convertframe(ctrlData)




