import pandas as pd
control = pd.read_csv('independentGenesControl.csv')
# case = pd.read_csv('independentGenesCase.csv')
ind_genes = list(control['0'])
# print ind_genes

dataFrame = pd.read_csv('independentcontrol.csv')

dep_genes = []
for item in dataFrame.iterrows():
     itemList = item[1]['antecedants'].split(',')
     for gene in itemList:
         # print item
         if gene in ind_genes:
             # print "gene: ", gene
             # print "consequents: ", item[1]['consequents']
            igenes = item[1]['consequents'].split(',')
            for xgene in igenes:
                dep_genes.append(xgene)



def sortAndUniq(input):
  output = []
  for x in input:
    if x not in output:
      output.append(x)
  output.sort()
  return output

dep_genesx = sortAndUniq(dep_genes)
print dep_genesx
df = pd.DataFrame(dep_genesx)
df.to_csv('dependentGenesControl.csv')



