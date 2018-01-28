# coding:utf-8
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier

a = np.array([[1, 2, 3, 4, 5, 6],[11,22,33,44,33,44]])
datas = pd.DataFrame(a)

def func(x):
    print "...."
    print x
    return sum(x)


print a



#print help(datas.apply)

