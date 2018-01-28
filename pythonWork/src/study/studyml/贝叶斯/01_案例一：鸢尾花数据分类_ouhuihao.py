#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False



path = 'datas/1.csv'  # 数据文件路径

data = pd.read_csv(path)
names = data.columns.values
X = data[names[:-1]]
print X
poly=PolynomialFeatures(degree=2)
X = poly.fit_transform(X)
print "*"*10
print X
#print help(PolynomialFeatures)
