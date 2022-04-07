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


for scale in prediction:  # 历遍三种尺度
    for bbox in scale:    # 历遍每个预测框
        obj = bbox[4]   # 获取该预测框的confidence（objectness）
        class_scores = bbox[:5]  # 获取该预测在coco数据集80个类别的概率
        class_id = np.argmax(class_scores)  # 获取概率类别的索引号
        class_name = classes[class_id]  # 获取概率最高类别的名称
        class_prob = class_scores[class_id]   # 获取概率最高类别的概率

        # 获取预测框中心点坐标，预测框宽高
        center_x = int(bbox[0] * width)
        center_y = int(bbox[1] * height)
        w = int(bbox[2] * width)
        h = int(bbox[3] * height)

        # 计算预测框左上角坐标
        x = int(center_x - w/2)
        y = int(center_y - h/2)

        # 将每个预测框的结果存放至上面列表中
        boxes.append([x, y, w, h])
        objectness.append(float(obj))
        class_ids.append(class_id)
        class_names.append(class_name)
        class_probs.append(class_prob)

print("boxes: ", boxes)
print("len(boxes): ", len(boxes))
print("len(objectness): ", len(objectness))


# 将预测框置信度objectness与个类别置信度class_pred相乘，获得最终该预测框置信度confidence
confidences = np.array(class_probs) * np.array(objectness)
print("len(confidences): ", len(confidences))

# plt.plot(objectness, label = "objectness")
# plt.plot(class_probs, label = "class_probs")
# plt.plot(confidences, label = "confidences")
# plt.legend()
# plt.show()



# 置信度过滤，非极大值抑制NMS
CONF_THRES = 0.1  # 制定置信度阈值，阈值越大，置信度过滤越强
NMS_THRES = 0.4  # 指定NMS阈值，阈值越小，NMS越强

indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRES, NMS_THRES)
print("index.flatten(): ", indexes.flatten())

print("len(indexes.flatten()): ", len(indexes.flatten()))

# 随即给每个预测框生成一种颜色
# colors = np.random.uniform(0, 255, size(len(boxes), 3))
colors = [[255, 0, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], \
            [255, 255, 0], [255, 0, 0], [100, 197, 28], [223, 155, 6], \
            [94, 218, 121], [139, 211, 142]]

# 历遍留下每一个框，可视化：
for i in indexes.flatten():

    # 获取坐标
    x, y, w, h = boxes[i]
    # 获取置信度
    confidence = str(round(confidences[i], 2))
    # 获取颜色， 画框
    color = colors[i % len(colors)]
    cv2.rectangle(img_input, (x,y), (x+w, y+h), color, 8)

    # 写类别名称和置信度
    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    print("i: ", i)
    print("class_name: ", class_names[i])
    strings = "{} {}".format(class_names[i], confidence)
    cv2.putText(img_input, strings, (x, y+20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)

look_img(img_input)
cv2.imwrite("/home/marco/result_yolo_img.jpg", img_input)

################ video ################
