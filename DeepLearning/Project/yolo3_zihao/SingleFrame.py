import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def look_img(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


net = cv2.dnn.readNet("/home/marco/yolov3.weights", "/home/marco/Estudiar/DeepLearning/Project/yolo3_zihao/yolov3.cfg")

print(type(net))
print(net)

coco_names_file = open("/home/marco/Estudiar/DeepLearning/Project/yolo3_zihao/coco.names", "r").read()
classes = coco_names_file.splitlines()
print("classes.shape: ", len(classes))

img_input = cv2.imread("/home/marco/Estudiar/DeepLearning/Project/yolo3_zihao/4.png")
# look_img(img_input)

height, width, channels = img_input.shape

print("img_input.shape: ", img_input.shape)
blob = cv2.dnn.blobFromImage(img_input, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
print("blob shape: ", blob.shape)

net.setInput(blob)
print("net.getLayerNames(): ", net.getLayerNames())
print("net.getUnconnectedOutLayers(): ", net.getUnconnectedOutLayers())

layersNames = net.getLayerNames()
print("layersNames: ", layersNames)

# output_layers_names = [layersNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
output_layers_names = ["yolo_82", "yolo_94", "yolo_106"]
print("output_layers_names: ", output_layers_names)

# 输入 yolo3神经网络，前向推断预测
prediction = net.forward(output_layers_names)


# 三个yolo3 三个尺度的输出结果
print("len(prediction): ",len(prediction))
print("prediction[0].shape: ", prediction[0].shape)
print("prediction[1].shape: ", prediction[1].shape)
print("prediction[2].shape: ", prediction[2].shape)

# 查看第二个尺度，索引为99的框的85维向量
print("prediction[1][99].shape: ", prediction[1][99].shape)
print("prediction[1][99]: ", prediction[1][99])



# 从三个尺度输出结果中解析所有预测框信息
# 存放预测框坐标
boxes = []

# 存放置信度
objectness = []

# 存放类别概率
class_probs = []

# 存放月册类别的索引号
class_ids = []

# 存放预测框类别名称
class_names = []


for scale in prediction:


