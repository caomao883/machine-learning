#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score
## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
## 读取数据
# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
path = './datas/iris.data'
data = pd.read_csv(path, header=None)
x, y = data[range(4)], data[4]
y = pd.Categorical(y).codes
x = x[[0, 1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=28, train_size=0.6)

clf = svm.SVC(C=1, kernel='linear')
clf.fit(x_train, y_train)

print clf.score(x_train, y_train)
print '训练集准确率：', accuracy_score(y_train, clf.predict(x_train))
print clf.score(x_test, y_test)
print '测试集准确率：', accuracy_score(y_test, clf.predict(x_test))
print '\npredict: ', clf.predict(x_train)

# 画图
N = 500
x1_min, x2_min = x.min()
x1_max, x2_max = x.max()

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)
grid_show = np.dstack((x1.flat, x2.flat))[0]

grid_hat = clf.predict(grid_show)
grid_hat = grid_hat.reshape(x1.shape)
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])

cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)
plt.show()