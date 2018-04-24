from fp_growth import find_frequent_itemsets

import csv
import pandas as pd
from  pymining import itemmining

transactions = []

with open('AssocationMatrixCtrl.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        # print row
        for item in row:
            # print item
            my_list = item.split(",")

            my_tuple = tuple(my_list)
            transactions.append(my_list)



tupleTransac = tuple(tuple(x) for x in transactions)
# print tupleTransac
# dataframe = pd.DataFrame(transactions)
# print dataframe

# print transactions
# for transaction in transactions:
#     print transaction
# with open
freqItemsets = []
report = find_frequent_itemsets(transactions, 4)
for itemset in report:
    freqItemsets.append(itemset)
print len(freqItemsets)
dataFrame = pd.DataFrame(freqItemsets)
dataFrame.to_csv('fpgrowth.csv')


# relim_input = itemmining.get_relim_input(tupleTransac)
# report = itemmining.relim(relim_input, min_support=2)
# print report