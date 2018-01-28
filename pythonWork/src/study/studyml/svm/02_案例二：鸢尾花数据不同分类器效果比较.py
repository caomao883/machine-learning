import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

## 读取数据
# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
path = './datas/iris.data'  # 数据文件路径
data = pd.read_csv(path, header=None)
x, y = data[range(4)], data[4]
y = pd.Categorical(y).codes
x = x[[0, 1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=28, train_size=0.6)
svm = SVC(C=1, kernel='linear')
svm.fit(x_train, y_train)

lr = LogisticRegression()
rc = RidgeClassifier()
knn = KNeighborsClassifier()

lr.fit(x_train, y_train)
rc.fit(x_train, y_train)
knn.fit(x_train, y_train)


svm_score1 = accuracy_score(y_train, svm.predict(x_train))
svm_score2 = accuracy_score(y_test, svm.predict(x_test))

lr_score1 = accuracy_score(y_train, lr.predict(x_train))
lr_score2 = accuracy_score(y_test, lr.predict(x_test))

rc_score1 = accuracy_score(y_train, rc.predict(x_train))
rc_score2 = accuracy_score(y_test, rc.predict(x_test))

knn_score1 = accuracy_score(y_train, knn.predict(x_train))
knn_score2 = accuracy_score(y_test, knn.predict(x_test))

## 画图
x_tmp = [0,1,2,3]
y_score1 = [svm_score1, lr_score1, rc_score1, knn_score1]
y_score2 = [svm_score2, lr_score2, rc_score2, knn_score2]

plt.figure(facecolor='w')
plt.plot(x_tmp, y_score1, 'r-', lw=2, label=u'训练集准确率')
plt.plot(x_tmp, y_score2, 'g-', lw=2, label=u'测试集准确率')
plt.xlim(0, 3)
plt.ylim(np.min((np.min(y_score1), np.min(y_score2)))*0.9, np.max((np.max(y_score1), np.max(y_score2)))*1.1)
plt.legend(loc = 'lower right')
plt.title(u'鸢尾花数据不同分类器准确率比较', fontsize=16)
plt.xticks(x_tmp, [u'SVM', u'Logistic', u'Ridge', u'KNN'], rotation=0)
plt.grid(b=True)
plt.show()
### 画图比较
N = 500
x1_min, x2_min = x.min()
x1_max, x2_max = x.max()

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)
grid_show = np.dstack((x1.flat, x2.flat))[0]

svm_grid_hat = svm.predict(grid_show)
svm_grid_hat = svm_grid_hat.reshape(x1.shape)

lr_grid_hat = lr.predict(grid_show)
lr_grid_hat = lr_grid_hat.reshape(x1.shape)

rc_grid_hat = rc.predict(grid_show)
rc_grid_hat = rc_grid_hat.reshape(x1.shape)

knn_grid_hat = knn.predict(grid_show)
knn_grid_hat = knn_grid_hat.reshape(x1.shape)

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w', figsize=(14,7))

plt.subplot(221)
plt.pcolormesh(x1, x2, svm_grid_hat, cmap=cm_light)
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)

plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.subplot(222)
plt.pcolormesh(x1, x2, lr_grid_hat, cmap=cm_light)
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花Logistic特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.subplot(223)
plt.pcolormesh(x1, x2, rc_grid_hat, cmap=cm_light)
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花Ridge特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.subplot(224)
plt.pcolormesh(x1, x2, knn_grid_hat, cmap=cm_light)
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花KNN特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.show()
