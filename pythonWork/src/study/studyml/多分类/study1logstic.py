#coding:utf-8
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time
def date_format(dt):
    import time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
path = 'datas\household_power_consumption.txt' ## 全部数据
path = 'datas\household_power_consumption_200.txt' ## 200行数据
path = 'datas\household_power_consumption_1000.txt' ## 1000行数据
df = pd.read_csv(path, sep=';', low_memory=False)

names=['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

new_df = df.replace('?', np.nan)
datas = new_df.dropna(how = 'any')
path = 'datas\household_power_consumption.txt' ## 全部数据
path = 'datas\household_power_consumption_200.txt' ## 200行数据
#path = 'datas\household_power_consumption_1000.txt' ## 1000行数据
df = pd.read_csv(path, sep=';', low_memory=False)

names=['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

new_df = df.replace('?', np.nan)
datas = new_df.dropna(how = 'any')
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]]


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, Y_train)


y_predict = lr.predict(X_test)


print "准确率:",lr.score(X_test, Y_test)


t=np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc = 'lower right')
plt.title(u"线性回归预测时间和功率之间的关系", fontsize=20)
plt.grid(b=True)
plt.show()
#PolynomialFeatures多项式，1,2,3,4
#fit_intercept表示是否要截距，既常数项
models = [
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', LinearRegression(fit_intercept=False))
    ])
]
model = models[0]

X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

t = np.arange(len(X_test))
N = 5
d_pool = np.arange(1, N, 1)
m = d_pool.size
clrs = []
for c in np.linspace(16711680, 255, m):
    clrs.append('#%06x' % c)
line_width = 3

plt.figure(figsize=(12, 6), facecolor='w')
for i, d in enumerate(d_pool):
    plt.subplot(N - 1, 1, i + 1)
    plt.plot(t, Y_test, 'r-', label=u'真实值', ms=10, zorder=N)
    model.set_params(Poly__degree=d)
    model.fit(X_train, Y_train)
    lin = model.get_params('Linear')['Linear']
    output = u'%d阶，系数为：' % d
    print output, lin.coef_.ravel()

    y_hat = model.predict(X_test)
    s = model.score(X_test, Y_test)

    z = N - 1 if (d == 2) else 0
    label = u'%d阶, 准确率=%.3f' % (d, s)
    plt.plot(t, y_hat, color=clrs[i], lw=line_width, alpha=0.75, label=label, zorder=z)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylabel(u'%d阶结果' % d, fontsize=12)

plt.legend(loc='lower right')
plt.suptitle(u"线性回归预测时间和功率之间的多项式关系", fontsize=20)
plt.grid(b=True)
plt.show()