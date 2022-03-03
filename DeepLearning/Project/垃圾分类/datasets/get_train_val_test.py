# 数据处理就不写了。
# 数据集地址：https://aistudio.baidu.com/aistudio/datasetdetail/30982

# 214个小类别
# 我们分为四个大类

import os
f_dir = "/home/agent/Estudiar/DeepLearning/Project/垃圾分类/datasets/"

ftr = open("/home/agent/Estudiar/DeepLearning/Project/垃圾分类/datasets/train1.txt", "w")
fval = open("/home/agent/Estudiar/DeepLearning/Project/垃圾分类/datasets/val1.txt", "w")
ftest = open("/home/agent/Estudiar/DeepLearning/Project/垃圾分类/datasets/test1.txt", "w")

for item in os.listdir("/home/agent/Estudiar/DeepLearning/Project/垃圾分类/datasets/"):
    if item == "train.txt":
        tmp_file = open(f_dir+item)
        for i in tmp_file:
            ftr.writelines("/home/agent/"+i)
    if item == "test.txt":
        tmp_file = open(f_dir+item)
        for i in tmp_file:
            ftest.writelines("/home/agent/"+i)
    if item == "val.txt":
        tmp_file = open(f_dir+item)
        for i in tmp_file:
            fval.writelines("/home/agent/"+i)
    