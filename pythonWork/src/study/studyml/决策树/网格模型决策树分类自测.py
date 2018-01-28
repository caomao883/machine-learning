from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


import pandas as pd
N = 500
M = 2
H = 2000
W = 20
x = (np.random.random((N,M))-0.5)*20
for i in range(N):
    x[i,1] = x[i,1]*H/W
y = np.random.randint(0,2,size=N)
def func(x):
    return (x + 9.) * x * (x - 9.)*3
for i in range(N):
    if(func(x[i][0])>x[i][1]):
        y[i]=0
    else:
        y[i]=1

# figure=plt.figure(figsize=(12,6))
# cm_dark = mpl.colors.ListedColormap(['r','b'])
# plt.scatter(x[:,0],x[:,1],c=y,cmap=cm_dark)
# plt.plot(np.arange(-10.,10.,step=0.1),func(np.arange(-10.,10.,step=0.1)),'r-')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


def getModel():
    pipe = Pipeline([
        # ("min", MinMaxScaler()),
        # ("skb", SelectKBest(chi2)),
        # ("pca", PCA()),
        ("desition", DecisionTreeClassifier())
    ])
    pams = {
        # "skb__k": [1,2,3,4],
        # "pca__n_components": [0.5,1.0],
        "desition__criterion": ['gini', "entropy"],
        "desition__max_depth": [2, 3, 4, 5, 6, 7, 8]
    }
    print(len(x_train), ",", len(y_train))
    gsc = GridSearchCV(pipe, param_grid=pams)
    return gsc
gsc = getModel()
model = gsc.fit(x_train, y_train)
print("best_params:", model.best_params_)
print("best_score:", model.best_score_)


plt.figure(figsize=(12,6))

print(plt.pcolormesh)
x = np.arange(start=-W/2,stop=W/2,step=0.1)
y = np.arange(start=-H/2,stop=H/2,step=0.1)
x,y = np.meshgrid(x,y)
xx = x.reshape((-1,1))
yy = y.reshape((-1,1))
dd = np.concatenate((xx,yy),axis=1)
print("dd.shape:",dd.shape)
y_predict = model.predict(dd)
Z = y_predict.reshape(x.shape)
print(Z.shape)
cm_dark = mpl.colors.ListedColormap(['r','b'])
plt.pcolormesh(x, y, Z, cmap=plt.cm.Paired)
plt.plot(np.arange(-W/2,W/2,step=0.1),func(np.arange(-W/2,W/2,step=0.1)),'r-')
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_dark)
plt.xlim(-W/2,W/2)
plt.ylim(-H/2,H/2)
plt.show()
print(y_train.shape)
