
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings

import sklearn
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def notEmpty(s):
    return s != ''

names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
path = "datas/boston_housing.data"
fd = pd.read_csv(path, header=None)
data = np.empty((len(fd), 14))
for i, d in enumerate(fd.values):
    d = map(float, filter(notEmpty, d[0].split(' ')))
    data[i] = d

x, y = np.split(data, (13,), axis=1)
y = y.ravel()

print "样本数据量:%d, 特征个数：%d" % x.shape
print "target样本数据量:%d" % y.shape[0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)

parameters = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 0.5],
    'gamma': [0.0001, 0.0005]
}
model = GridSearchCV(SVR(), param_grid=parameters, cv=3)
model.fit(x_train, y_train)

print "最优参数列表:", model.best_params_
print "最优模型:", model.best_estimator_
print "最优准确率:", model.best_score_

print "训练集准确率:%.2f%%" % (model.score(x_train, y_train) * 100)
print "测试集准确率:%.2f%%" % (model.score(x_test, y_test) * 100)

## 画图
colors = ['g-', 'b-']
ln_x_test = range(len(x_test))
y_predict = model.predict(x_test)

plt.figure(figsize=(16,8), facecolor='w')
plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'真实值')
plt.plot(ln_x_test, y_predict, 'g-', lw = 3, label=u'SVR算法估计值,$R^2$=%.3f' % (model.best_score_))
plt.legend(loc = 'upper left')
plt.grid(True)
plt.title(u"波士顿房屋价格预测(SVM)")
plt.xlim(0, 101)
plt.show()


