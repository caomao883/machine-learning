import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LogisticRegressionCV,LinearRegression
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)


path = "datas/breast-cancer-wisconsin.data"
names = ['id','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
         'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei',
        'Bland Chromatin','Normal Nucleoli','Mitoses','Class']

df = pd.read_csv(path, header=None,names=names)

datas = df.replace('?', np.nan).dropna(how = 'any')
datas.head(5)


X = datas[names[1:10]]
Y = datas[names[10]]


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=0)


ss = StandardScaler()
X_train = ss.fit_transform(X_train) ## 训练模型及归一化数据


lr = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty='l2', solver='lbfgs', tol=0.01)
lr.fit(X_train, Y_train)


r = lr.score(X_train, Y_train)
print "R值（准确率）：", r
print "稀疏化特征比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100)
print "参数：",lr.coef_
print "截距：",lr.intercept_


from sklearn.externals import joblib

joblib.dump(ss, "models/logistic/ss.model")
joblib.dump(lr, "models/logistic/lr.model")


from sklearn.externals import joblib
oss = joblib.load("models/logistic/ss.model")
olr = joblib.load("models/logistic/lr.model")


X_test = oss.transform(X_test)

Y_predict = olr.predict(X_test)


x_len = range(len(X_test))
plt.figure(figsize=(14,7), facecolor='w')
plt.ylim(0,6)
plt.plot(x_len, Y_test, 'ro',markersize = 8, zorder=3, label=u'真实值')
plt.plot(x_len, Y_predict, 'go', markersize = 14, zorder=2, label=u'预测值,$R^2$=%.3f' % olr.score(X_test, Y_test))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'乳腺癌类型', fontsize=18)
plt.title(u'Logistic回归算法对数据进行分类', fontsize=20)
plt.show()

print Y_test
print Y_predict
