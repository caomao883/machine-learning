import matplotlib as mpl
import matplotlib.pyplot as plt
#图片数据
from sklearn import datasets, svm, metrics

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
#数字图片加载
digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print("分类器%s的分类效果:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("混淆矩阵为:\n%s" % metrics.confusion_matrix(expected, predicted))

plt.figure(facecolor='gray', figsize=(12,5))
images_and_predictions = list(zip(digits.images[n_samples / 2:][expected != predicted], expected[expected != predicted], predicted[expected != predicted]))
for index, (image,expection, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'预测值/实际值:%i/%i' % (prediction, expection))
images_and_predictions = list(zip(digits.images[n_samples / 2:][expected == predicted], expected[expected == predicted], predicted[expected == predicted]))
for index, (image,expection, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 6)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'预测值/实际值:%i/%i' % (prediction, expection))

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()

