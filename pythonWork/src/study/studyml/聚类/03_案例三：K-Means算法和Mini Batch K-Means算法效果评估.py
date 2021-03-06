import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

centers = [[1, 1], [-1, -1], [1, -1]]
clusters = len(centers)

X, Y = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7, random_state=28)

k_means = KMeans(init='k-means++', n_clusters=clusters, random_state=28)
t0 = time.time()
k_means.fit(X)
km_batch = time.time() - t0
print "K-Means算法模型训练消耗时间:%.4fs" % km_batch

batch_size = 100
mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=batch_size, random_state=28)
t0 = time.time()
mbk.fit(X)
mbk_batch = time.time() - t0
print "Mini Batch K-Means算法模型训练消耗时间:%.4fs" % mbk_batch

km_y_hat = k_means.labels_
mbkm_y_hat = mbk.labels_

k_means_cluster_centers = k_means.cluster_centers_
mbk_means_cluster_centers = mbk.cluster_centers_
print "K-Means算法聚类中心点:\ncenter=", k_means_cluster_centers
print "Mini Batch K-Means算法聚类中心点:\ncenter=", mbk_means_cluster_centers
order = pairwise_distances_argmin(k_means_cluster_centers,
                                  mbk_means_cluster_centers)

### 效果评估
score_funcs = [
    metrics.adjusted_rand_score,
    metrics.v_measure_score,
    metrics.adjusted_mutual_info_score,
    metrics.mutual_info_score,
]

## 2. 迭代对每个评估函数进行评估操作
for score_func in score_funcs:
    t0 = time.time()
    km_scores = score_func(Y, km_y_hat)
    print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (score_func.__name__, km_scores, time.time() - t0))

    t0 = time.time()
    mbkm_scores = score_func(Y, mbkm_y_hat)
    print(
    "Mini Batch K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs\n" % (score_func.__name__, mbkm_scores, time.time() - t0))


