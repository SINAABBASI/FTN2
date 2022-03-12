from os import sep
from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt

ch = [0.34, 0.85, 0.34]
ch = np.array(ch)
n = 5 #mf
dim = 3
mid = len(ch)//2
bit= np.zeros((n,1))
o =  np.zeros((dim,2**n))
color = []
c_points = []
for i in range(2**n):
    for j in range(n):
        if ((2**j)&i)!= 0: bit[j] = 1
        else: bit[j] = -1

    for j in range(mid,n-mid):
        o[j-1][i] = np.round(bit[j-1]*ch[0] + bit[j]*ch[1] + bit[j+1]*ch[2],3)

    color  += ['red' if (bit[1] > 0) else 'blue']
    c_points += [(o[0][i],o[1][i],o[2][i],color[-1])]

    # exit()

print(len(c_points))
c_points = set(c_points)
print(*c_points,sep='\n')
print(len(c_points))

fig = plt.figure()
ax = fig.add_subplot(
    projection='3d'
    )
ax.scatter(o[0],o[1],o[2],c=color)

plt.show()