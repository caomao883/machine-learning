import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm, datasets

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
#加载数据，并且获取前两列作为特征属性
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

def my_kernel(x, y):
    """
    We create a custom kernel:

                 (2  0)
    k(x, y) = x  (    ) y.T
                 (0  1)
    """
    M = np.array([[2, 0], [0, 1]])
    return np.dot(np.dot(x, M), y.T)

clf = svm.SVC(kernel=my_kernel)
clf.fit(X, Y)

score = clf.score(X,Y)
print "训练集准确率:%.2f%%" % (score * 100)

h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w', figsize=(8,4))
plt.pcolormesh(xx, yy, Z, cmap=cm_light)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_dark)

plt.title(u'SVM中自定义核函数')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.text(x_max - .3, y_min + .3, (u'准确率%.2f%%' % (score * 100)).lstrip('0'),  size=15, horizontalalignment='right')
plt.show()

