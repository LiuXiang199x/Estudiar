# pytorch-yolov3

#### 介绍
pytorch实现yolov3目标检测模型
B站手撸代码视频地址：https://www.bilibili.com/video/BV1Mh411p7Pw?spm_id_from=333.999.0.0

#### 软件架构
pytorch+yolov3


#### 安装教程

1.  装新版的库不会报错，pytorch>=1.9

#### 使用说明

1.  数据集图片存放地址  data/images   vocxml文件存放地址：data/image_voc
2.  将数据集存放好之后运行 make_data_txt.py生成data.txt文件
3.  使用k-means聚类自己的数据集，将结果在config.py中进行替换
4.  运行train.py开始训练
5.  运行test.py进行测试


#### 视频地址

B站手撸代码地址：https://www.bilibili.com/video/BV1Mh411p7Pw?spm_id_from=333.999.0.0