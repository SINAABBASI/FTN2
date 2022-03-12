import numpy as np
import math as math
pi = np.arccos(-1)

def RRC(time,T,beta):
    a= T/4/beta
    out = np.zeros(len(time))
    for j in range(0,len(time)):
        if time[j] == 0:
            out[j] = 1/np.sqrt(T)*(1-beta+4*beta/pi)
        elif (a==time[j]) | (-a==time[j]): 
            out[j] = beta/np.sqrt(2*T)*((1+2/pi)*np.sin(pi/4/beta)+(1-2/pi)*np.cos(pi/4/beta))
        else:
            out[j] = (np.sin(pi*(1-beta)*time[j]/T)+4*beta*time[j]/T*np.cos(pi*(1+beta)*time[j]/T))/(pi*time[j]/np.sqrt(T)*(1-(4*beta*time[j]/T)**2))
    return out


def downsample(a,sep):
    out = np.zeros((len(a)+sep)//sep)
    # print(len(a),sep)
    for i in range(len(a)):
        if i%sep == 0:
            out[i//sep] = a[i]
    return out

def upsample(a,sep):
    out = np.zeros(len(a)*sep)
    for i in range(len(a)):
        out[i*sep] = a[i]
        for j in range(1,sep): out[i*sep+j] = 0
    return out 