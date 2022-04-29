from os import sep
from random import randrange, seed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
# from xgboost import XGBRegressor
n = 1000
h = np.load(f'H6Length{n}.npy')
h_sp = h[2][0:5]
# h_sp = h[3][0:7]
# ch_pw = 
# ch_h = np.random.normal(0,np.sqrt(chPw/2),(n,n))
# print(h[3][0:7])

class cluster:
    def __init__(self,n_tran=19):
        self.n_tran = n_tran
        self.mid_tran = self.n_tran//2
        self.trun_len = 5
        self.mid_trun = self.trun_len//2
    
    def fitModel(self,neigh=1):
        pos = []
        neg = []
        # cnt = 0
        points = set()
        X = []
        XX = []
        Y = []
        YY = []
        for i in range(2**self.n_tran):
            temp = 0
            sig = []
            for j in range(self.n_tran):
                if (i&(2**j)) != 0:
                    sig += [1]
                else: 
                    sig += [-1]
            o = []
            Y += [sig[self.mid_tran]]
            for j in range(self.mid_trun,self.n_tran - self.mid_trun):
                o += [np.round(np.dot(sig[j-self.mid_trun:j+self.mid_trun+1],h_sp),4)]
            X += [o]
            points.add(tuple(o)) 
            

            if sig[self.mid_tran] < 0: 
                neg += [o[len(o)//2]]
            else:
                pos += [o[len(o)//2]]
            

            temp = np.round(temp,4)
        

        print("number of cluster points:", len(points))
        X = np.array(X)
        Y = np.array(Y)
        XX = np.array(XX)
        YY = np.array(YY)
        pos = np.unique(pos)
        neg = np.unique(neg)
        print(*pos)
        # print(cnt)
        print("-*******-")
        print(*neg)



        print('cluster points shape',X.shape)
        self.model = KNeighborsClassifier(n_neighbors=neigh,weights='distance').fit(X,Y)
        # self.model = MLPClassifier(random_state=1, max_iter=300).fit(X,Y)
        print('error in training:', sum(self.model.predict(X) != Y))
        return self.model

    def reduced_model(self):
        
        return
    
    
    def testModel(self,ebno,num_iteration):
        noisePw = 1/(10**(ebno/10))
        dim = self.n_tran - self.trun_len + 1
        err = []
        for _ in range(num_iteration):
            s = 2*np.random.randint(2,size=(n,1))-1
            noise = np.sqrt(noisePw/2)*np.random.standard_normal((n,1))
            r = np.dot(h,s) + noise
            X_t = np.zeros((n-(self.n_tran-1),dim))
            Y = []
            for i in range(self.mid_tran,n-self.mid_tran):
                for id,val in enumerate(r[i-dim//2:i+dim//2+1]):
                    X_t[i-self.mid_tran][id] = np.round(val,4)
                Y += [1 if s[i] > 0 else -1]

            Y = np.array(Y)
            Y_t = self.model.predict(X_t)
            print(np.unique(Y_t))
            err += [sum(Y_t != Y)/len(Y)]
        print(np.mean(err))
        return np.mean(err)


# model = SVC(gamma=2, C=2).fit(X,Y)


# rng = range(3,15,2)
# Y = []
# for i in rng:
#     sysModel = cluster(n_tran=i)
#     sysModel.fitModel()
#     Y += [sysModel.testModel(ebno=6,num_iteration=100)]

# plt.plot(rng,Y)
# plt.show()


# rng = range(1,6,2)
# Y = []
# for i in rng:
#     sysModel = cluster()
#     sysModel.fitModel(neigh=i)
#     Y += [sysModel.testModel(ebno=6,num_iteration=100)]

# plt.plot(rng,Y)
# plt.show()


# rng = range(4,13,2)
# num_it = [10,10,100,100,1000]
rng = [12]
num_it = [1000]
Y = []
sysModel = cluster()
sysModel.fitModel()
for i,v in zip(rng,num_it):
    Y += [sysModel.testModel(ebno=i,num_iteration=v)]

uncoded6_SD = [0.0404, 0.017024793388429754, 0.002578268876611418, 0.00026040995230114736, 5.66213743e-06]
# uncoded5_SD = [0.07172414, 0.03961905, 0.00739927, 0.00101771, 5.36256457e-05]

# array([0.0816]), array([0.04938272]), array([0.00764818]), array([0.00138072])
# [array([0.08416667]), array([0.03162791]), array([0.00793651]), array([0.00103977])]

plt.yscale("log")
plt.plot(rng,Y,'rx-')
plt.plot(rng,uncoded6_SD,'bo-')
plt.ylabel('BER')
plt.xlabel('eb/no')
plt.show()



## 13 observation
# tau = 0.6
# 0.051219512195121955
# 0.016880081300813005
# 0.004278455284552845
# 0.0004085365853658537
# 2.7439024390243903e-05

# tau = 0.5
# 0.09470468431771893
# 0.04675152749490835
# 0.013951120162932791
# 0.0026680244399185336
# 0.00026272912423625254



## 15 observation
# tau = 0.6
# 0.057433808553971476
# 0.018228105906313647
# 0.0032790224032586564
# 0.0003971486761710794
# 3.1568228105906315e-05

# tau = 0.5
# 0.09010204081632653
# 0.04408163265306122
# 0.012030612244897957
# 0.0019183673469387757
# 0.00017448979591836735



