from formatData import processData
import pandas as pd
import matplotlib.pyplot as plt

dataPath = '/Users/macuser/Downloads/514hw1csv.csv'

train_data, test_data = processData(dataPath)

# train_data.groupby('TEMP').sum()
# test_data.groupby('TEMP').sum()
# plt.hist(train_data.TEMP, alpha=0.5,label='Train Data')
# plt.hist(test_data.TEMP, alpha=0.5 ,label='Test Data')
# plt.legend(loc='upper right')
# plt.title('Temp Dis
# tribution Train/Test Data')
# plt.show()
# plt.title('Humidity Distribution in Training Data')
# plt.xlim(xmin=40, xmax = 110)
# plt.ylim(ymin=0, ymax=12)
# train_data.groupby('HUM').sum()
# plt.hist(train_data.HUM, bins=15)
# plt.show()

# plt.title('Cloud Distribution in Training Data')
# plt.xlim(xmin=0, xmax = 110)
# plt.ylim(ymin=0, ymax=14)
# train_data.groupby('CLOUD').sum()
# plt.hist(train_data.CLOUD, bins=15)
# plt.show()

# plt.title('Temp Distribution in Training Data')
# # plt.xlim(xmin=0, xmax = 110)
# # plt.ylim(ymin=0, ymax=14)
# train_data.groupby('TEMP').sum()
# plt.hist(train_data.TEMP, bins=15)
# plt.show()

plt.title('Light Distribution in Training Data')

train_data.groupby('LIGHT').sum()
plt.hist(train_data.LIGHT, bins=15)
plt.show()





