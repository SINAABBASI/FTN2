from random import seed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
# from xgboost import XGBRegressor


def min_dist(X,Y):
    X1 = []
    X2 = []
    X = X.tolist()
    mdist = 100
    for x,y in zip(X,Y):
        if y > 0: X1 += [x]
        else: X2 += [x]
    X1 = np.array(X1)
    X2 = np.array(X2)
    for x1 in X1:
        for x2 in X2:
            # print(x1-x2)
            # exit()
            mdist = min(mdist,np.sqrt(sum([i**2 for i in x1-x2])))

    print(mdist)


n = 1000
ebno = 6
h = np.load(f'H6Length{n}.npy')
noise = np.load(f'noise{ebno}len{n}.npy')
s = np.load(f'dataLen{n}.npy')
# h = np.round(h,1)
print(h[0][:10])

# s = 2*np.random.randint(2,size=(n,1))-1
# np.save(f'dataLen{n}',s)
# exit()

# es = sum([i**2 for i in s]) / n
# noisePw = es/(10**(ebno/10))
# noise = np.sqrt(noisePw/2)*np.random.standard_normal((n,1))
# np.save(f'noise{ebno}len{n}',noise)
# exit()
dim = 11
mid = (dim//2)
# print(mid)
r = np.dot(h,s) 
X = np.zeros((n-2*mid,dim))
Y = []
for i in range(mid,len(r)-mid):
    for j in range(dim):
        X[i-mid][j] = r[i-mid + j]
    Y += [1 if s[i] > 0 else -1]
# exit()
#classification
# model = SVC(gamma=2, C=2)
# model = XGBRegressor(n_estimators=500, learning_rate=0.01, n_jobs=10)
# model = GaussianProcessClassifier(1.0 * RBF(1.0))
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X,Y)

#clustering
# spc = KMeans(n_clusters=2, random_state=0).fit(X)
# spc = GaussianMixture(n_components=2, random_state=0).fit(X)
# Y = np.array(Y)
# Y = 2*Y - 1
# print(Y)

min_dist(X,Y)
print(sum(model.predict(X) != Y))
# exit()
r = r + noise
o = np.zeros((dim,n-2*mid))
X_t = np.zeros((n-2*mid,dim))
color = []
for i in range(mid,len(r)-mid):
    for j in range(dim):
        X_t[i-mid][j] = r[i-mid + j]
        o[j][i-mid] = r[i-mid + j]
    # color += ['red' if s[i] > 0 else 'blue']


# print(spc.labels_)
# exit()
Y_t = model.predict(X_t)
for i,v in enumerate(Y_t):
    if abs(v-1) > abs(v+1): Y_t[i] = -1
    else: Y_t[i] = 1
print(Y_t)
# Y_t = 2*Y_t - 1

err = 0
for i in range(mid,len(r)-mid):
    if s[i] != Y_t[i-mid]: err += 1
print(err)
print(sum(Y_t != Y))

# fig = plt.figure()
# ax = fig.add_subplot(
#     projection='3d'
#     )
# ax.scatter(o[mid - 1],o[mid],o[mid + 1],c=Y_t)

# plt.show()

noisePw = 1/(10**(ebno/10))
    
err = []
for i in range(100):
    s = 2*np.random.randint(2,size=(n,1))-1
    # noise = np.sqrt(noisePw/2)*np.random.standard_normal((n,1))
    r = np.dot(h,s) + noise
    o = np.zeros((dim,n-2*mid))
    X_t = np.zeros((n-2*mid,dim))
    Y = []
    for i in range(mid,len(r)-mid):
        for j in range(dim):
            X_t[i-mid][j] = r[i-mid + j]
            o[j][i-mid] = r[i-mid + j]
        Y += [1 if s[i] > 0 else -1]
    Y_t = model.predict(X_t)
    print(sum(Y_t != Y))
    err += [sum(Y_t != Y)/1000]
print(np.mean(err))

# 78 3d
# 82 2d
# 108 1d
