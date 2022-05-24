import numpy as np
import math


def LLR(n, candList, noisePw, dis):
    ### LLR calculation
    la = np.zeros((n,1))
    for i in range(n):
        one = 0
        mone = 0
        for c in candList:
            if c[i] == 1: one += 1
            else: mone += 1
        # print(one,mone,len(candList))
   
        if one == 0: la[i] = -10
        elif mone == 0: la[i] = 10
        else: la[i] = math.log(one/len(candList)) - math.log(mone/len(candList))
    
    # print(la)
    # print(answer)
    ld = np.zeros((n,1))
    cMod = np.zeros((n-1,1))
    laMod = np.zeros((n-1,1))
    for i in range(n):
        one = 0
        mone = 0
        cnt = 0
        for c in candList:
            likelihoodP = math.exp(-1 / noisePw * dis[cnt]**2) 
            id = 0
            for j in range(n):
                if j == i:continue
                cMod[id] = c[j]
                laMod[id] = la[j]
                id += 1
            if c[i] == 1:
                one += likelihoodP*math.exp(0.5*np.dot(cMod.T,laMod))
            else:
                mone += likelihoodP*math.exp(0.5*np.dot(cMod.T,laMod))
            cnt += 1
        if one == 0: ld[i] = la[i] - 10
        elif mone == 0: ld[i] = la[i] + 10
        else: ld[i] = la[i] + math.log(one) - math.log(mone)
    
    
    return ld
