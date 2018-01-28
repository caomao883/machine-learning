import numpy as np
import pandas
from sklearn.metrics.pairwise import pairwise_distances_argmin

dist = pairwise_distances_argmin([[1, 2],[4,3]], [[100, 20],[4,3],[3,4]])
print dist

