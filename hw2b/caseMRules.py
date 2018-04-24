from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import numpy as np

dataframe = pd.read_csv('ARMatrixControl.csv')
del dataframe['Classes']
del dataframe['Unnamed: 0']

min_support=0.095
print "support: ", min_support
frequent_itemsets = apriori(dataframe, min_support=min_support, use_colnames=True)
fileName = "control_frequent_itemsets_" + str(min_support) + '.csv'
frequent_itemsets.to_csv(fileName)

for confidence_increment in range(50, 100, 10):
    min_confidence = confidence_increment / float(100)
    print "confidence: ", min_confidence
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    fileName = "control_association_rules_C" + str(min_confidence) + "_S" + str(min_support)
    rules.to_csv(fileName+'.csv')


print "DONE BITCH"










