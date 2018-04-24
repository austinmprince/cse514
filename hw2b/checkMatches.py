import pandas as pd

control = pd.read_csv('independentGenesControl.csv')
case = pd.read_csv('independentGenesCase.csv')
count = 0
match = []
for cont in control['0']:
    for cas in case['0']:
        if cont == cas:
            count += 1
            match.append(cont)
print count
print match


