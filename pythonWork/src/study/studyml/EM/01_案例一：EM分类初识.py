#coding:utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.mixture import GaussianMixture

# 解决中文显示问题


mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# %matplotlib tk


## 使用scikit携带的EM算法或者自己实现的EM算法
def trainModel(style, x):
    if style == 'sklearn':
        # 对象创建
        g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000, init_params='kmeans')
        # 模型训练
        g.fit(x)
        # 效果输出
        print '类别概率:\t', g.weights_[0]
        print '均值:\n', g.means_, '\n'
        print '方差:\n', g.covariances_, '\n'
        mu1, mu2 = g.means_
        sigma1, sigma2 = g.covariances_
        return (mu1, mu2, sigma1, sigma2)
    else:
        num_iter = 100
        n, d = data.shape

        mu1 = data.min(axis=0)
        mu2 = data.max(axis=0)
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        pi = 0.5

        for i in range(num_iter):
            # E Step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            tau1 = pi * norm1.pdf(data)
            tau2 = (1 - pi) * norm2.pdf(data)
            gamma = tau1 / (tau1 + tau2)

            # M Step
            mu1 = np.dot(gamma, data) / np.sum(gamma)
            mu2 = np.dot((1 - gamma), data) / np.sum((1 - gamma))
            sigma1 = np.dot(gamma * (data - mu1).T, data - mu1) / np.sum(gamma)
            sigma2 = np.dot((1 - gamma) * (data - mu2).T, data - mu2) / np.sum(1 - gamma)
            pi = np.sum(gamma) / n

            # 输出信息
            j = i + 1
            if j % 10 == 0:
                print j, ":\t", mu1, mu2

        # 效果输出
        print '类别概率:\t', pi
        print '均值:\t', mu1, mu2
        print '方差:\n', sigma1, '\n\n', sigma2, '\n'

        # 返回结果
        return (mu1, mu2, sigma1, sigma2)

np.random.seed(28)
N = 500
M = 250

mean1 = (0, 0, 0)
cov1 = np.diag((1, 2, 3))
data1 = np.random.multivariate_normal(mean1, cov1, N)

mean2 = (2, 2, 1)
cov2 = np.array(((1, 1, 3), (1, 2, 1), (0, 0, 1)))
data2 = np.random.multivariate_normal(mean2, cov2, M)

data = np.vstack((data1, data2))

y1 = np.array([True] * N + [False] * M)
y2 = ~y1

style = 'sklearn'
style = 'self'
mu1, mu2, sigma1, sigma2 = trainModel(style, data)

norm1 = multivariate_normal(mu1, sigma1)
norm2 = multivariate_normal(mu2, sigma2)
tau1 = norm1.pdf(data)
tau2 = norm2.pdf(data)


dist = pairwise_distances_argmin([mean1, mean2], [mu1, mu2], metric='euclidean')
print "距离:", dist
if dist[0] == 0:
    c1 = tau1 > tau2
else:
    c1 = tau1 < tau2
c2 = ~c1

acc = np.mean(y1 == c1)
print u'准确率：%.2f%%' % (100*acc)


## 画图
fig = plt.figure(figsize=(12, 6), facecolor='w')

ax = fig.add_subplot(121,projection='3d')
ax.scatter(data[y1, 0], data[y1, 1], data[y1, 2], c='r', s=30, marker='o', depthshade=True)
ax.scatter(data[y2, 0], data[y2, 1], data[y2, 2], c='g', s=30, marker='^', depthshade=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(u'原始数据', fontsize=16)

ax = fig.add_subplot(122, projection='3d')
ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(u'EM算法分类', fontsize=16)

plt.suptitle(u'EM算法的实现,准备率：%.2f%%' % (acc * 100), fontsize=20)
plt.subplots_adjust(top=0.90)
plt.tight_layout()
plt.show()