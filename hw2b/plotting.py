import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('controlARCount.csv')

y = list(df['Support'])
x = list(df['Confidence'])
z = list(df['Num Items'])
y1 = y[:4]
y2 = y[5:9]
y3 = y[10:14]
y4 = y[15:19]
y5 =  y[20:24]
x1 = x[:4]
x2 = x[5:9]
x3 = x[10:14]
x4 = x[15:19]
x5 =  x[20:24]
z1 = z[:4]
z2 = z[5:9]
z3 = z[10:14]
z4 = z[15:19]
z5 =  z[20:24]

fig = plt.figure()
ax = fig.gca(projection='3d')
# for i in range(1, 6):
#     x = 'x' + str(i)
#     y = 'y' + str(i)
#     z = 'z' + str(i)
#     print x, y, z
ax.plot(x1, y1, z1, label='parametric curve')
ax.plot(x2, y2, z2, label='parametric curve')
ax.plot(x3, y3, z3, label='parametric curve')
ax.plot(x4, y4, z4, label='parametric curve')
ax.plot(x5, y5, z5, label='parametric curve')
# ax.plot(x, y, z, label='parametric curve')
ax.legend(xlabel='Min Support', ylabel='Min Confidence', zlabel='Log(#Association Rules')
#
plt.show()
