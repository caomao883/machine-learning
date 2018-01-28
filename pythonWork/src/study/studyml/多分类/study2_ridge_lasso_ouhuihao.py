#coding:utf-8
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
np.random.seed(100)
np.set_printoptions(linewidth=1000, suppress=True)
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8*x**3 + x**2 - 14*x - 7 + np.random.randn(N)
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
x.shape = -1,1
y.shape = -1,1
models = [
    Pipeline([
        ("Poly",PolynomialFeatures()),
        ("Linear",LinearRegression(fit_intercept=False))
    ]),
    Pipeline([
        ("Poly",PolynomialFeatures()),
        ("Linear",RidgeCV(alphas=np.logspace(-3,2,50),fit_intercept=False))]),
     Pipeline([
        ("Poly",PolynomialFeatures()),
        ("Linear",LassoCV(alphas=np.logspace(-3,2,50),fit_intercept=False))]),
    Pipeline([
        ("Poly",PolynomialFeatures()),
        ("Linear",ElasticNetCV(alphas=np.logspace(-3,2,50),l1_ratio=[0.1,0.5,0.7,0.9,1.],fit_intercept=False))
    ])
]

model = models[0]
degree = np.arange(1, N, 4)  # 阶
print degree
dm = degree.size
for i, d in enumerate(degree):
    plt.subplot(int(np.ceil(dm / 2.0)), 2, i + 1)
    plt.plot(x,y,'ro',zorder=i)
    model.set_params(Poly__degree=9)
    model.fit(x,y)
    y_pre= model.predict(x)
    scores = model.score(x,y)
    lin = model.get_params('Linear')['Linear']
    output = u'%d阶，系数为：' % (d)
    print output, lin.coef_.ravel()
    plt.title(u"%d阶，准确率:%.3f"%(9,scores))
    plt.plot(x,y_pre,'r-')

    x_hat = np.linspace(x.min(), x.max(), num=100)
    x_hat.shape = -1, 1
    y_hat = model.predict(x_hat)
    s = model.score(x, y)

    z = N - 1 if (d == 2) else 0
    label = u'%d阶, 正确率=%.3f' % (d, s)
    plt.plot(x_hat, y_hat, alpha=0.75, label=label, zorder=z)

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
plt.suptitle(u'线性回归过拟合显示', fontsize=22)
plt.show()



# plt.figure(facecolor='w')
# degree = np.arange(1, N, 4)  # 阶
# dm = degree.size
# colors = []  # 颜色
# for c in np.linspace(16711680, 255, dm):
#     colors.append('#%06x' % c)
#
# model = models[0]
# for i, d in enumerate(degree):
#     plt.subplot(int(np.ceil(dm / 2.0)), 2, i + 1)
#     plt.plot(x, y, 'ro', ms=10, zorder=N)
#
#     model.set_params(Poly__degree=d)
#
#     model.fit(x, y.ravel())
#
#     lin = model.get_params('Linear')['Linear']
#     output = u'%d阶，系数为：' % (d)
#     print output, lin.coef_.ravel()
#
#
#     x_hat = np.linspace(x.min(), x.max(), num=100)
#     x_hat.shape = -1, 1
#     y_hat = model.predict(x_hat)
#     s = model.score(x, y)
#
#     z = N - 1 if (d == 2) else 0
#     label = u'%d阶, 正确率=%.3f' % (d, s)
#     plt.plot(x_hat, y_hat, color=colors[i], lw=2, alpha=0.75, label=label, zorder=z)
#
#     plt.legend(loc='upper left')
#     plt.grid(True)
#     plt.xlabel('X', fontsize=16)
#     plt.ylabel('Y', fontsize=16)
#
# plt.tight_layout(1, rect=(0, 0, 1, 0.95))
# plt.suptitle(u'线性回归过拟合显示', fontsize=22)
# plt.show()