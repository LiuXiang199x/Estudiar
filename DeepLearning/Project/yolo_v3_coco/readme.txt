工程顺序：
	1. 标注数据（jason，voc，txt）
	2. 配置文件（coco是一个80分类，如果我们只识别人，那就一个类别而以）,脚本文件sh创建网络层参数
	
	3. 标签格式转换（因为比如说labelme等不同工具使用的工具标出的结果是x1，y1，x2，y2；但是yolo用的是中心点Cx，Cy，w，h ——> x和y是一个相对位置，取值范围0-1）
	4. 创建训练/测试数据和标签路径文件。(转为yolo对应的数据格式)
		4.1 创建yolo数据格式 class x_, y_, w_, h_
		4.2 生成classes_names.txt
		4.3 生成训练，测试的图片路径txt

	----------------------------------- 准备好了所有数据集和格式 -----------------------------------------------

	5. train.txt, val.txt准备好代码:
		5.1 --model_def config/yolov3-custom.cfg
			--data_config config/custom.data
			--pretrained_weights weights/darknet53.conv.74


