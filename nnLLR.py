from turtle import shape
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as spio

n_ch = 1000
h_ch = np.load(f'H6Length{n_ch}.npy')
ch_len = 5
h_sp = h_ch[ch_len//2][0:ch_len] 

h = np.zeros((ch_len,ch_len))

for i in range(ch_len//2+1):
    h[i][:ch_len//2 + i + 1] = h_sp[ch_len//2 - i:]
for i in range(ch_len//2+1):
    h[ch_len//2+i] = h[ch_len//2-i][::-1]

print(h)


class cluster:
    def __init__(self,n_tran=17):
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
            

            # if sig[self.mid_tran] < 0: 
            #     neg += [o[len(o)//2]]
            # else:
            #     pos += [o[len(o)//2]]
            

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
        # self.model = GaussianNB().fit(X,Y)
        print('error in training:', sum(self.model.predict(X) != Y))
        return self.model

    def reduced_model(self):
        
        return
    
    

    def testModelLLR(self,ebno,data):
        noisePw = 1/(10**(ebno/10))
        dim = self.n_tran - self.trun_len + 1
        for i in range(0,len(data),n_ch):
            s = data[i:i+n_ch]
            s = np.array(s).reshape((n_ch, 1))
            noise = np.sqrt(noisePw/2)*np.random.standard_normal((n_ch,1))
            r = np.dot(h_ch,s) + noise
            X_t = np.zeros((n_ch-(self.n_tran-1),dim))
            Y = []
            for i in range(self.mid_tran,n_ch-self.mid_tran):
                for id,val in enumerate(r[i-dim//2:i+dim//2+1]):
                    X_t[i-self.mid_tran][id] = np.round(val,4)
                Y += [1 if s[i] > 0 else -1]
            Y = np.array(Y)
            # Y_t = self.model.predict(X_t)
            Y_t = self.model.kneighbors(X_t)

            return Y_t

    def testModel(self,ebno,num_iteration):
        noisePw = 1/(10**(ebno/10))
        dim = self.n_tran - self.trun_len + 1
        err = []
        for _ in range(num_iteration):
            print(_)
            s = 2*np.random.randint(2,size=(n_ch,1))-1
            noise = np.sqrt(noisePw/2)*np.random.standard_normal((n_ch,1))
            r = np.dot(h_ch,s) + noise
            X_t = np.zeros((n_ch-(self.n_tran-1),dim))
            Y = []
            for i in range(self.mid_tran,n_ch-self.mid_tran):
                for id,val in enumerate(r[i-dim//2:i+dim//2+1]):
                    X_t[i-self.mid_tran][id] = np.round(val,4)
                Y += [1 if s[i] > 0 else -1]
            Y = np.array(Y)
            
            Y_t = self.model.predict(X_t)
            err += [sum(Y_t != Y)/len(Y)]
        print(np.mean(err))
        return np.mean(err)


mat = spio.loadmat('./data/testData5e3_int.mat')
dataIn = np.array(mat['intrl'],dtype=int)[0]
# print(len(dataIn))

sysModel = cluster()
sysModel.fitModel(neigh=8)

y = sysModel.testModelLLR(4,dataIn)
print(y)