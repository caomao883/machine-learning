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
path = 'datas\household_power_consumption_1000.txt' ## 1000行数据
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

#fit_intercept表示是否要截距，既常数项
print "..................................."
modes = [Pipeline([('Poly',PolynomialFeatures()),('Linear',LinearRegression(fit_intercept=False))])]
mode = modes[0]
X = datas[names[0:2]]
X = X.apply(lambda x:pd.Series(date_format(x)),axis=1)
Y = datas[names[4]]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
t = np.arange(len(Y_test))
plt.figure(figsize=(12,6))
N = 2
for i in range(1,N):
    plt.subplot(N,1,i)
    plt.xlabel(u"%d横坐标"%i)
    plt.ylabel(u'%d阶'%i)
    mode.set_params(Poly__degree=i)
    mode.fit(X_train,Y_train)
    line = mode.get_params('Linear')['Linear']
    print "%d阶系数为:"%i ,line.coef_.ravel()
    y_predic = mode.predict(X_test)
    score = mode.score(X_test, Y_test)
    label= u"第%d阶，准确率：%f"%(i,score)
    plt.plot(t, Y_test, '-r', label=u'真实值')
    plt.plot(t,y_predic,'-b',label=label)
    plt.legend(loc='lower right')

plt.grid(True)
plt.show()

print lr.coef_
