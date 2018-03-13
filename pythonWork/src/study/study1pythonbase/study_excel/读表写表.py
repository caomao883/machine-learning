#coding:utf-8
import pandas as pd

import numpy as np
#表的路径
basestation_end = "result/result.xlsx"
#读表
data = pd.read_excel("data/test.xlsx",header=None)
#将表输出到终端
print(data)
#写表
data.to_excel(basestation_end,header=None,index=None)