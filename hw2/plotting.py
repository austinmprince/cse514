import matplotlib.pyplot as plt
import csv
import numpy as np

# dict = {'GI_11056053-S': 0.0027882157261658012, 'GI_20127571-S': 0.0050398331957721629,
#         'GI_19743835-S': 0.0039704410581217651, 'GI_11321616-S': 0.0028956815509122887,
#         'GI_21071055-S': 0.0027196866847077508, 'GI_21166356-S': 0.0056737506219214136,
#         'GI_18375674-I': 0.019904465817223337, 'GI_20143911-A': 0.0026771261361705524,
#         'GI_11386200-S': 0.0032752732596487491, 'GI_14702170-I': 0.002690158066130789,
#         'GI_10835070-S': 0.003336572028160005, 'GI_19923366-S': 0.003319209625265281,
#         'GI_20270388-S': 0.0027466107285091072, 'GI_11037060-A': 0.0026316675417878467,
#         'GI_11968044-S': 0.0039257096178841253, 'GI_20336758-S': 0.0040625853582450609,
#         'GI_20544178-S': 0.0028730940054034673, 'GI_14149858-S': 0.0068523408765699496,
#         'GI_13899314-S': 0.005310486433459845, 'GI_11496880-S': 0.00563107219211691,
#         'GI_10835229-S': 0.0064054341394356512, 'GI_20544150-I': 0.0063323045455964651,
#         'GI_20546504-S': 0.0055570801353173318, 'GI_13375790-S': 0.0092531469706814036,
#         'GI_19923286-S': 0.005709021405139735, 'GI_14249537-S': 0.032080026126367102,
#         'GI_20149303-S': 0.0032307450918880672, 'GI_20127596-S': 0.036905861443115237,
#         'GI_16306579-S': 0.002812236486792383, 'GI_17105397-I': 0.0026269243827337979}


# print len(dict)
def barPlot(dict, frac, features=True):
        n_groups = len(dict)
        fix, ax = plt.subplots()
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14)
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.4

        rects = plt.bar(index, dict.values(), bar_width, alpha=opacity, color='b', label='Freq')
        if features == True:
                string = "Features"
        else:
                string = "Instances"
        frac = frac*100
        print features
        print frac
        plt.xlabel('Gene')
        plt.ylabel('Significance')
        plt.title('Gene Significance with ' + str(int(frac)) + '% of ' + string )
        plt.xticks(index + bar_width / 2, list(dict.keys()), rotation=90)
        plt.tight_layout()
        plt.show()



# with open('dict.csv', 'wb') as csv_file:
#     writer = csv.writer(csv_file)
#     for key, value in dict.items():
#        writer.writerow([key, value])