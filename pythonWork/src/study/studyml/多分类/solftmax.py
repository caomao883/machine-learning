#coding:utf-8
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
#读取数据
path1 = "datas/winequality-red.csv"
df1 = pd.read_csv(path1, sep=";")
df1['type'] = 1 #设置数据类型为红葡萄酒

path2 = "datas/winequality-white.csv"
df2 = pd.read_csv(path2, sep=";")
df2['type'] = 2 #设置数据类型是白葡萄酒

#合并二个df
df = pd.concat([df1,df2], axis=0)

#自变量
names = ["fixed acidity","volatile acidity","citric acid",
         "residual sugar","chlorides","free sulfur dioxide",
         "total sulfur dioxide","density","pH","sulphates",
         "alcohol", "type"]
#因变量名称
quality = "quality"

#显示数据，方便自己查看，可以不做
df.head(5)
#异常数据处理
new_df = df.replace('?', np.nan)
datas = new_df.dropna(how = 'any')# 只要有列为空，就进行删除操作
print "原始数据条数:%d；异常数据处理后数据条数:%d；异常数据条数:%d" % (len(df), len(datas), len(df) - len(datas))

#提取自变量和因变量
X = datas[names]
Y = datas[quality]

#对数据分割-训练数据和测试数据分割
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

print "训练数据条数%d；数据特征个数:%d；测试数据条数:%d" % (X_train.shape[0], X_train.shape[1], X_test.shape[0])

#数据归一化
ss = MinMaxScaler()
X_train = ss.fit_transform(X_train) #对训练数据--训练并转换--标准化的转换

#构建模型并训练模型
lr = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100),
                          multi_class='multinomial', penalty='l2', solver='lbfgs')
lr.fit(X_train, Y_train)

#模型效果提取
r = lr.score(X_train, Y_train)
print "R值：", r
print "特征稀疏化比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100)
print "参数：",lr.coef_
print "截距：",lr.intercept_

#对测试数据标准化
X_test = ss.transform(X_test)
#对测试数据进行预测
Y_predict = lr.predict(X_test)

#图表的展示
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

#查看数据分布情况
#df[df.qulity==i]会列出所有quality为i的行
[len(df[df.quality == i]) for i in range(11)]

#对数据进行降维处理后建模，查看效果，使用PCA降维（有时候进行特征抽取和降维对于模型有可能是没有多大的改进的）
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Normalizer

#进行分割
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X,Y,test_size=0.25,random_state=0)
print "训练数据条数:%d；数据特征个数:%d；测试数据条数:%d" % (X1_train.shape[0], X1_train.shape[1], X1_test.shape[0])

#数据标准化（归一化）
ss2 = Normalizer()
X1_train = ss2.fit_transform(X1_train)

# #特征选择
#skb = SelectKBest(chi2,k=8)#特征数目可以自己指定，也可以用默认的
#X1_train = skb.fit_transform(X1_train,Y_train) #训练模型及特征选择

# #降维
# pca=PCA(n_components=5)#将样本数据维度降低到指定的维度个数
# X1_train = pca.fit_transform(X1_train)


#构建模型并训练模型
lr2 = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100),
                          multi_class='multinomial', penalty='l2', solver='lbfgs')
lr2.fit(X1_train, Y1_train)


r = lr2.score(X1_train, Y1_train)
print "R值：", r
#ravel表示多维数组，降为一位，[[1,2],[3,4]]变为[1,2,3,4]
#a.array([1,2,3,4,0,0])==0 最后得到一个boolean序列
print "特征稀疏化比率：%.2f%%" % (np.mean(lr2.coef_.ravel() == 0) * 100)
print "参数：",lr2.coef_
print "截距：",lr2.intercept_


#对于测试数据进行处理--归一化、特征选择、降维
X1_test = ss2.transform(X1_test)
#X1_test = skb.transform(X1_test)
#X1_test = pca.fit_transform(X1_test) #

#应用前面训练的模型对测试数据进行预测
Y1_predict = lr2.predict(X1_test)

#画图操作
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

#从AUC角度看模型效果
from sklearn.preprocessing import label_binarize
from sklearn import metrics
y_test_hot = label_binarize(Y_test,classes=(3,4,5,6,7,8,9)).ravel()

#计算原始数据模型
#得到预测的损失值
lr_y_score = lr.decision_function(X_test).ravel()
#计算ROC
lr_fpr, lr_tpr, lr_threasholds = metrics.roc_curve(y_test_hot,lr_y_score)
#计算AUC
lr_auc = metrics.auc(lr_fpr, lr_tpr)

#计算降维后的数据模型
#decision_function 的值等于X1_test乘以特征系数
lr2_y_score = lr2.decision_function(X1_test).ravel()
#计算ROC
lr2_fpr, lr2_tpr, lr2_threasholds = metrics.roc_curve(y_test_hot,lr2_y_score)
#计算AUC
lr2_auc = metrics.auc(lr2_fpr, lr2_tpr)

print "原始数据AUC值:", lr_auc
print "降维数据AUC值:", lr2_auc