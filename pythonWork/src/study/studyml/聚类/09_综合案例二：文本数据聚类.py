from time import time
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans


### 加载模拟数据
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
print u'加载的20新闻数据中的数据类别为:',categories

dataset = fetch_20newsgroups(data_home='datas', subset='all', categories=categories,
                             shuffle=True, random_state=42)
print("%d条数据；%d个新闻类别" % (len(dataset.data), len(dataset.target_names)))

labels = dataset.target

target_cluster_k = np.unique(labels).shape[0]
features = 2 ** 20
components = 5
mini_batch_km_batchsize = 1000

hasher1 = HashingVectorizer(n_features=features, stop_words='english', non_negative=True,
                            norm=None, binary=False, token_pattern=u'(?u)\\b\\w\\w+\\b')
tt = TfidfTransformer(norm='l2', use_idf=True)
hasher2 = HashingVectorizer(n_features=features, stop_words='english', non_negative=False,
                            norm='l2', binary=False, token_pattern=u'(?u)\\b\\w\\w+\\b')
tv = TfidfVectorizer(max_df=0.5, max_features=features, min_df=2, stop_words='english', use_idf=True)

vectorizers = [
    ('hashing&tf-idf', make_pipeline(hasher1, tt), False),
    ('hasing', make_pipeline(hasher2), False),
    ('tf-idf', make_pipeline(tv), True)
]

svd = TruncatedSVD(n_components=components)
normalizer = Normalizer(norm='l2', copy=False)
sn = make_pipeline(svd, normalizer)

mbkm = MiniBatchKMeans(n_clusters=target_cluster_k, init='k-means++', n_init=5,
                       init_size=10 * mini_batch_km_batchsize, batch_size=mini_batch_km_batchsize)
km = KMeans(n_clusters=target_cluster_k, init='k-means++', max_iter=100, n_init=5)
cluster_als = [('Mini-Batch-KMeans', mbkm), ('KMeans', km)]

for vectorizer_name, vectorizer, can_inverse in vectorizers:
    print "============================================"
    print "采用'%s'的方式将文本数据转换为特征矩阵" % vectorizer_name

    t0 = time()
    X = vectorizer.fit_transform(dataset.data)
    print "转换消耗时间:%.3fs" % (time() - t0)
    print "样本数量:%d,特征属性数量:%d" % X.shape

    t0 = time()
    X = sn.fit_transform(X)
    print "SVD分解及归一化消耗时间:%.3fs" % (time() - t0)
    print "降维&归一化操作后，样本数量:%d,特征属性数量:%d" % X.shape

    for cluster_name, cluster_al in cluster_als:
        print
        print "使用算法%s对数据进行建模操作" % cluster_name
        t0 = time()
        cluster_al.fit(X)
        print "模型构建消耗时间:%.3fs" % (time() - t0)
        print "%s算法效果评估相关系数" % cluster_name
        print(u"均一性/同质性: %0.3f" % metrics.homogeneity_score(labels, cluster_al.labels_))
        print("完整性: %0.3f" % metrics.completeness_score(labels, cluster_al.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels, cluster_al.labels_))
        print("Adjusted Rand-Index(ARI): %.3f" % metrics.adjusted_rand_score(labels, cluster_al.labels_))
        print("轮廓系数: %0.3f" % metrics.silhouette_score(X, cluster_al.labels_, sample_size=1000))
        print "聚类中心点为:", cluster_al.cluster_centers_

        if can_inverse:
            print "获取文本转换特征矩阵中，各个分类考虑特征属性的前10个feature特征（10个单词）："
            original_space_centroids = svd.inverse_transform(cluster_al.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
            terms = vectorizer.named_steps.items()[0][1].get_feature_names()
            for i in range(target_cluster_k):
                print "类别%d:" % i,
                for ind in order_centroids[i, :10]:
                    print ' %s' % terms[ind],
                print
    print
    print
print "==================算法完成======================"


