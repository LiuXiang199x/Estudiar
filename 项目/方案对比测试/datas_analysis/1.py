from 项目.RL_ROS.datas_analysis.FBE_datasets import Datasets

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


a,b = Datasets().get_sum()
print(a)
print(b)
tmp = []
for i in range(10):
    tmp.append(b[i]/a[i]*100/5)

print(tmp)
print(10000*0.05/10)
print(260000*0.05*0.05/10)