import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn import metrics


mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)


path1 = "datas/winequality-red.csv"
df1 = pd.read_csv(path1, sep=";")
df1['type'] = 1

path2 = "datas/winequality-white.csv"
df2 = pd.read_csv(path2, sep=";")
df2['type'] = 2


df = pd.concat([df1,df2], axis=0)


names = ["fixed acidity","volatile acidity","citric acid",
         "residual sugar","chlorides","free sulfur dioxide",
         "total sulfur dioxide","density","pH","sulphates",
         "alcohol", "type"]

quality = "quality"


df.head(5)



new_df = df.replace('?', np.nan)
datas = new_df.dropna(how = 'any')
print "原始数据条数:%d；异常数据处理后数据条数:%d；异常数据条数:%d" % (len(df), len(datas), len(df) - len(datas))


X = datas[names]
Y = datas[quality]


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

print "训练数据条数:%d；数据特征个数:%d；测试数据条数:%d" % (X_train.shape[0], X_train.shape[1], X_test.shape[0])


ss = MinMaxScaler()
X_train = ss.fit_transform(X_train)


lr = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100),
                          multi_class='multinomial', penalty='l2', solver='lbfgs')
lr.fit(X_train, Y_train)


r = lr.score(X_train, Y_train)
print "R值：", r
print "特征稀疏化比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100)
print "参数：",lr.coef_
print "截距：",lr.intercept_


X_test = ss.transform(X_test)

Y_predict = lr.predict(X_test)


x_len = range(len(X_test))
plt.figure(figsize=(14,7), facecolor='w')
plt.ylim(-1,11)
plt.plot(x_len, Y_test, 'ro',markersize = 8, zorder=3, label=u'真实值')
plt.plot(x_len, Y_predict, 'go', markersize = 12, zorder=2, label=u'预测值,$R^2$=%.3f' % lr.score(X_train, Y_train))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'葡萄酒质量', fontsize=18)
plt.title(u'葡萄酒质量预测统计', fontsize=20)
plt.show()


[len(df[df.quality == i]) for i in range(11)]


from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Normalizer


X1_train,X1_test,Y1_train,Y1_test = train_test_split(X,Y,test_size=0.025,random_state=0)
print "训练数据条数:%d；数据特征个数:%d；测试数据条数:%d" % (X1_train.shape[0], X1_train.shape[1], X1_test.shape[0])


ss2 = Normalizer()
X1_train = ss2.fit_transform(X1_train)



lr2 = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100),
                          multi_class='multinomial', penalty='l2', solver='lbfgs')
lr2.fit(X1_train, Y1_train)


r = lr2.score(X1_train, Y1_train)
print "R值：", r
print "特征稀疏化比率：%.2f%%" % (np.mean(lr2.coef_.ravel() == 0) * 100)
print "参数：",lr2.coef_
print "截距：",lr2.intercept_



X1_test = ss2.transform(X1_test)
#X1_test = skb.transform(X1_test)
#X1_test = pca.fit_transform(X1_test) #


Y1_predict = lr2.predict(X1_test)


x1_len = range(len(X1_test))
plt.figure(figsize=(14,7), facecolor='w')
plt.ylim(-1,11)
plt.plot(x1_len, Y1_test, 'ro',markersize = 8, zorder=3, label=u'真实值')
plt.plot(x1_len, Y1_predict, 'go', markersize = 12, zorder=2, label=u'预测值,$R^2$=%.3f' % lr2.score(X1_train, Y1_train))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'葡萄酒质量', fontsize=18)
plt.title(u'葡萄酒质量预测统计(降维处理)', fontsize=20)
plt.show()



from sklearn.preprocessing import label_binarize
from sklearn import metrics
y_test_hot = label_binarize(Y_test,classes=(3,4,5,6,7,8,9)).ravel()


lr_y_score = lr.decision_function(X_test).ravel()

lr_fpr, lr_tpr, lr_threasholds = metrics.roc_curve(y_test_hot,lr_y_score)

lr_auc = metrics.auc(lr_fpr, lr_tpr)


lr2_y_score = lr2.decision_function(X1_test).ravel()

lr2_fpr, lr2_tpr, lr2_threasholds = metrics.roc_curve(y_test_hot,lr2_y_score)

lr2_auc = metrics.auc(lr2_fpr, lr2_tpr)

print "原始数据AUC值:", lr_auc
print "降维数据AUC值:", lr2_auc



