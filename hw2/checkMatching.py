import pandas as pd

dataFrame = pd.read_csv('testMatch.csv')
print dataFrame

column1 = dataFrame.ix[:,0].values
column2 = dataFrame.ix[:,1].values
column3 = dataFrame.ix[:,2].values
count = 0
for value in column1:
    for svalue in column2:
        for tvalue in column3:
            if value == svalue and value == tvalue:
                count += 1

print count
