import pandas as pd

dataFrame = pd.read_csv('independentcontrol.csv')

antecedent_list = []
for ant in dataFrame['antecedants']:
    ants = ant.split(',')
    for item in ants:
        antecedent_list.append(item)


consequent_list = []
for con in dataFrame['consequents']:
    cons = con.split(',')
    for item in cons:
        consequent_list.append(item)




def sortAndUniq(input):
  output = []
  for x in input:
    if x not in output:
      output.append(x)
  output.sort()
  return output

sortedans = sortAndUniq(antecedent_list)
# print sortedans

sortedcons = sortAndUniq(consequent_list)
# print sortedcons

uniqList = []
for item in sortedans:
    count = 0
    for citem in sortedcons:
        if item == citem:
            count += 1
    if count == 0:
        uniqList.append(item)
    count = 0

df = pd.DataFrame(uniqList)
df.to_csv('independentGenesControl.csv')




