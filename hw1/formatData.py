import pandas as pd


# dataPath = '/Users/macuser/Downloads/514hw1csv.csv'
#
# allData = pd.read_csv(dataPath)
#
#
# allData.loc[(allData.TINE > 0, 'TINE')] = 1
# allData.loc[(allData.TINE <= 0, 'TINE')] = 0
#
# allData.dropna(how='any', inplace=True)
# allData.to_csv("editedHW1.csv", index=False)

def processData(dataPath):

    CSV_COLUMNS = ['DATE', 'TIME', 'TEMP', 'HUM', 'LIGHT', 'CLOUD']

    train_data = pd.read_csv(dataPath, names=CSV_COLUMNS, nrows=28, skiprows=1)
    test_data = pd.read_csv(dataPath, names=CSV_COLUMNS, skiprows=29)

    train_data.dropna(how='any', inplace=True)
    train_data.loc[(train_data.TIME > 0, 'TIME')] = 1
    train_data.loc[(train_data.TIME <= 0, 'TIME')] = 0
    test_data.loc[(test_data.TIME > 0, 'TIME')] = 1
    test_data.loc[(test_data.TIME <= 0, 'TIME')] = 0

    return train_data, test_data


















