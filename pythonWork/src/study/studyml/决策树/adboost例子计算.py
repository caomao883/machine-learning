#coding:utf-8
from sklearn import metrics
import numpy as np
import numpy as np
from sklearn import metrics

def sig(x):
    x=(-x+1)/2
    return x
def minEoor(w, y):
    result = 1.0
    result_g = y
    for i in range(11):
        g = np.append(np.linspace(1,1,i),np.linspace(-1,-1,10-i))
        error = sum(w*sig(g*y))
        if error < result:
            result = error
            result_g = g
    for i in range(11):
        g = np.append(np.linspace(-1,-1,i),np.linspace(1,1,10-i))
        error = sum(w*sig(g*y))
        if error < result:
            result = error
            result_g = g
    return result, result_g

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
def culaccuracy(f, y):
    count = 0
    predict = sigmoid(f) > 0.5
    y = (y==1)
    # print "预测值:",predict
    # print "真实值:",y
    for y_predict, true_y in zip(predict, y):
        if y_predict == true_y:
            count += 1
    return 1.*count / len(y)



def adaboost(W,Y):
    f = np.linspace(0.,0.,len(Y))
    m = 0
    while True:
        m+=1
        error1,G1 = minEoor(W,Y)
        a1 = 0.5*np.log((1-error1)/error1)

        Z1 = sum(W*np.exp(-a1*G1*Y))
        W1 = W*np.exp(-a1*G1*Y)/Z1
        f1 = f+a1*G1

        print ""
        print "最小错误率:",error1
        print "G%d:"%m,G1
        print "G%d:系数:"%m,a1
        print "f%d:"%m, f1
        print "权重W%d:"%m,W1
        accuracy = culaccuracy(f1, Y)
        print "第%d次迭代准确率:%.2f%%"%(m, accuracy*100)
        if accuracy  == 1.0 or m > 20:
            break

        W = W1
        f = f1






X = np.array([0,1,2,3,4,5,6,7,8,9])
Y = np.array([1, 1, 1, -1, -1, -1., 1, 1, 1, -1])
W0 = np.linspace(0.1,0.1, len(Y))
adaboost(W0,Y)




