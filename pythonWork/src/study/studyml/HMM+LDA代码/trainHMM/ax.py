#coding:utf-8

import math
import  matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.05, 3, 0.05)
y1 = [math.pow(i, 2) for i in x]
y2 = [math.pow(2, i) for i in x]
y3 = [math.log(i, 2) for i in x]
plt.plot(x, y1, linewidth=2, color='b', label='$x^2$')
plt.plot(x, y2, linewidth=2, color='g', label='$2^x$')
plt.plot(x, y3, linewidth=2, color='r', label='$log2(x)$')
plt.plot([1], [1], 'bo')
plt.plot([1], [0], 'ro')
plt.legend(loc='lower right')
plt.xlim(0, 3)
plt.grid(True)
plt.show()
