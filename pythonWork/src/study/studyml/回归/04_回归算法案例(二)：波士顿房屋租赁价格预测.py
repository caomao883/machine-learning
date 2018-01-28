import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

def notEmpty(s):
    return s != ''


mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)


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


models = [
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', RidgeCV(alphas=np.logspace(-3,1,20)))
        ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', LassoCV(alphas=np.logspace(-3,1,20)))
        ])
]


parameters = {
    "poly__degree": [3,2,1],
    "poly__interaction_only": [True, False],
    "poly__include_bias": [True, False],
    "linear__fit_intercept": [True, False]
}


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

titles = ['Ridge', 'Lasso']
colors = ['g-', 'b-']
plt.figure(figsize=(16, 8), facecolor='w')
ln_x_test = range(len(x_test))

plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'真实值')
for t in range(2):
    model = GridSearchCV(models[t], param_grid=parameters, n_jobs=1)

    model.fit(x_train, y_train)

    print "%s算法:最优参数:" % titles[t], model.best_params_
    print "%s算法:R值=%.3f" % (titles[t], model.best_score_)

    y_predict = model.predict(x_test)

    plt.plot(ln_x_test, y_predict, colors[t], lw=t + 3, label=u'%s算法估计值,$R^2$=%.3f' % (titles[t], model.best_score_))

plt.legend(loc='upper left')
plt.grid(True)
plt.title(u"波士顿房屋价格预测")
plt.show()


model = Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
            ('linear', LassoCV(alphas=np.logspace(-3,1,20), fit_intercept=False))
        ])

model.fit(x_train, y_train)



print model.get_params('linear')['linear'].coef_
print "参数:", zip(names,model.get_params('linear')['linear'].coef_)
print "截距:", model.get_params('linear')['linear'].intercept_


name = names
x, x2 = np.split(data, (13,), axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, x2, test_size=0.2, random_state=0)

ss = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=True, interaction_only=True)
# include_bias: 决定多项式转换后是否添加常数项，默认为true，添加
# interaction_only：决定多项式转换是否仅仅添加不同item之间的结合数据/交互项，
# False表示否，会添加类似a^2；true表示仅仅添加类似a1*a2
linear = LassoCV(alphas=np.logspace(-3,1,20), fit_intercept=False)


x_train = ss.fit_transform(x_train, y_train)
print x_train[0]

x_train = poly.fit_transform(x_train, y_train)
print x_train[0]
powers = poly.powers_
print powers.shape
print poly.powers_

linear.fit(x_train, y_train)
coef = linear.coef_
print coef
print coef.shape

# name = ['a', 'b', 'c']
name2 = []
for i in powers:
    tname = zip(name, i)
    tname = filter(lambda x: x[1] != 0, tname)
    tname = map(lambda x: x[0] * x[1], tname)
    if tname:
        tname = reduce(lambda x,y: x+"_"+y, tname)
    else:
        tname = "1"
    name2.append(tname)


print zip(name2, coef)

len(name2)


