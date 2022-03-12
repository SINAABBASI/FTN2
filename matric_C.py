# from matplotlib import pyplot as plt
# from RRC import RRC,downsample,upsample
import numpy as np



# trn = 6
# ts = 0.1
# Th = 1
# T = 0.6
# tOverTs = 6
# t = [i/10 for i in range(int(-trn/ts),int(trn/ts)+1)]

# rollOfFac = 0.35
# rollOfFac2 = 0.2
# rrc = RRC(t,Th,rollOfFac)
# rrc2 = RRC(t,T,rollOfFac2)

# c = downsample(rrc,tOverTs)
# c = upsample(c,tOverTs)
# c = c[0:-(tOverTs-1)]

# # c = c / np.linalg.norm(c)
# c = c * np.sqrt(T)
# co  = np.convolve(c,rrc2)
# co = co[int(trn/ts):int(trn/ts)+len(c)]

# # ax2.plot(t,co,'r-')
# # ax2.plot(t,rrc,'b--')
# # ax2.set_xlabel('(b)')
# # plt.show()




# l = 1000
# plt.stem(c)
# plt.show()
# c = downsample(c,tOverTs)
# print(len(c))
# print(np.round(c,3))
# v = c[int(len(c)/2):]
# print(v)
# C = np.zeros((l,l))
# for i in range(l):
#     for j in range(l):
#         if abs(j-i) < len(v):
#             C[i][j] = v[abs(j-i)]
#         else:
#             C[i][j] = 0

# np.save("H"+str(trn)+"Length"+str(l),C)
# print(np.round(C,2))
l=1000
v = [1, 0.5]
C = np.zeros((l,l))
for i in range(l):
    for j in range(l):
        if abs(j-i) < len(v):
            C[i][j] = v[abs(j-i)]
        else:
            C[i][j] = 0

np.save("H"+str(2)+"Length"+str(l),C)
print(np.round(C,2))