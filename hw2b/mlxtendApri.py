from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

dataframe = pd.read_csv('ARMatrixCase.csv')
del dataframe['Classes']
del dataframe['Unnamed: 0']
# print dataframe

print "Case 8s, 50c"
frequent_itemsets = apriori(dataframe, min_support=0.08, use_colnames=True)
print frequent_itemsets
frequent_itemsets.to_csv('8S50Ccaseitemsset.csv')
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
print rules
rules.to_csv('8S50Ccaserules.csv')




