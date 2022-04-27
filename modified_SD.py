import scipy.io as spio
import numpy as np
import csv
from matplotlib import pyplot as plt
import numpy as np
import math as math
INF = 1000111000111
# H = np.identity(25) #no ISI
H = np.load("./C6Length25.npy")
n = m = len(H[0])
mOrder = 2
# mat = spio.loadmat('./trainData/trainData10e6.mat')
# dataIn = np.array(mat['convTrain'],dtype=int)[0]

# dataMod = []
# ###bpsk modulation
# for i in range(0,len(dataIn)):
#     dataMod += [dataIn[i]*2-1]
# dataMod = np.array(dataMod,dtype=int)
# dataMod = dataMod.reshape(int(len(dataIn)/math.log2(mOrder)/n),n,1)
# print(dataMod.shape)




his = []
cnadNum = 4
ber = []
tmp = []

rng = [8]
for ebno in rng:
    calc_std = 0.39297375479619
    calc_mean = 1.303061741132164
    err = 0
    print(ebno,calc_mean,calc_std)
    it=0
    # while True :
    # for it in range(len(dataMod)):
    it+=1
    if (err >= 50 or it>10**7): 
        it -= 1
        break
    alpha = 1
    s= 2 * np.random.random_integers(1,mOrder,(m,1))- (mOrder + 1)        
    total_energy = sum([ii**2 for ii in s])
    energyPerBit = total_energy / n
    noisePw = energyPerBit/(10**(ebno/10))
    v = np.sqrt(noisePw/2)*np.random.standard_normal((n,1)) ## scale in np.random.normal equals to standard deviation   
    initialSig = s.copy()
    x = np.dot(H,s) + v
    d = calc_mean**2

    for sp in range(0,25,5):
        print(s[sp:sp+5])
        print()
    exit()


    q1 = np.zeros((n,m),dtype='complex')
    res = np.linalg.qr(H)
    R = res[1]
    q1 = res[0]
    y = np.dot(q1.conj().T,x)
    _y = y.copy()
    D = np.zeros(m)
    UB = np.zeros(m)
    k = m - 1
    D[k] = np.sqrt(d)
    setUB = 1 #should the upperbond be delcared or not
    ans = INF
    answer = np.zeros(m)
    

    nCand = []
    ###Start
    while True :
        k = m - 1
        _y = y.copy()
        D = np.zeros(m)
        UB = np.zeros(m)
        D[k] = np.sqrt(d)
        setUB = 1
        while True :
            if setUB == 1:
                if (D[k] + _y[k]) * R[k][k] > (-D[k] + _y[k]) * R[k][k]  : 

                    UB[k] = np.floor((D[k] + _y[k]) / R[k][k])
                    s[k] = np.ceil((-D[k] + _y[k]) / R[k][k])  - 1  
                else :
                    UB[k] = np.floor((-D[k] + _y[k]) / R[k][k])
                    s[k] = np.ceil((D[k] + _y[k]) / R[k][k])  - 1

                te = s[k] + 1
                for j in range(mOrder - 1, -mOrder , -2):
                    if te > j : 
                        break
                    s[k] = j - 2
            s[k] = s[k] + 2
            # print(k,s[k],UB[k])
            setUB = 0
            if s[k] <= UB[k] and s[k] < mOrder:
                if k == 0 :
                    if ans > np.linalg.norm(np.dot(H,s)-x):
                        ans = np.linalg.norm(np.dot(H,s)-x)
                        answer = s.copy()
                    if len(nCand) == cnadNum:
                        nCand += [np.linalg.norm(np.dot(H,s)-x)]
                        nCand = sorted(nCand)
                        nCand.pop()
                        D[m-1] = nCand[-1]
                    else:
                        nCand += [np.linalg.norm(np.dot(H,s)-x)]
                else :
                    k = k - 1
                    _y[k] = y[k]
                    for i in range(k+1,m) :
                        _y[k] -= (R[k][i] * s[i])
                    D[k] = np.sqrt(D[k+1]**2 - (_y[k+1] - R[k+1][k+1] * s[k+1])**2)
                    setUB = 1
                continue
            else : 
                k = k + 1
                if k == m :
                    break

        if ans == INF:
            print("upd")
            d = (calc_mean + alpha * calc_std)**2
            alpha += 1
            nCand = []
        else :
            break
    
    nCand = sorted(nCand)
    # if ( nCand[0] != np.linalg.norm(np.dot(H,answer)-x)):print("WRONG",)
    ####################
    his.append(nCand[-1])
    for i in range(0,n):
        if answer[i] != initialSig[i]:
            print(err)
            err = err + abs(answer[i] - initialSig[i]) // 2
    it += 1
    print(ebno, err,err/(it*n))
    ber += [err/(it*n)]
    # print(np.std(his),np.mean(his))
print(ber)
pb_theory = [0.5*math.erfc(np.sqrt(10**(i/10))) for i in rng]
print(pb_theory)
plt.yscale("log")
plt.xlabel("Eb/N0")
plt.ylabel("BER")
plt.plot(rng,ber,'gx--')
plt.plot(rng,pb_theory)
plt.legend(["noISI_uncoded_implVerification","Theory"])
plt.grid(True, which="both", ls="-")
plt.show()

## 4-10 theory 
###[0.012500818040737556, 0.0023882907809328075, 0.00019090777407599314, 3.872108215522037e-06]

### 4-10 tau0.7 
# [array([0.02222222]), array([0.00473934]), array([0.00069235]), array([9.375e-06])]
