from cProfile import label
from importlib import import_module
import json
import os

name2id = {"person": 0, "cloud": 1}

# convert xml/jason points coord to yolo's data(x_, y_, w_, h_)
def yolo_convert(img_size, box):

    # (img_w, img_h) (x1, y1, x2, y2)
    x_ = (box[0] + box[2]) / (2*img_size[0])
    y_ = (box[1] + box[3]) / (2*img_size[1])
    w_ = (box[2] - box[0]) / img_size[0]
    h_ = (box[3] - box[1]) / img_size[1]

    return (x_, y_, w_, h_)

def decode_json(json_folder_path, json_name):

    txt_name = "/home/marco/Estudiar/DeepLearning/Project/yolo_v3_coco/datas/simpleDatasets/labelme/txt/" + json_name[0] + ".txt"
    txt_file = open(txt_name, "w")

    json_path = os.path.join(json_folder_path, json_name)
    data = json.load(open(json_path, "r", encoding="gb2312"))

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    for item in data["shapes"]:
        label_name = item['label']
        if (item['shape_type'] == "rectangle"):
            x1 = int(item['points'][0][0])
            y1 = int(item['points'][0][1])
            x2 = int(item['points'][1][0])
            y2 = int(item['points'][1][1])

            boudingbox = (x1, y1, x2, y2)
            yolo_bbox = yolo_convert((img_w, img_h), boudingbox)

            txt_file.write(str(name2id[label_name]) + \
                " " + str(yolo_bbox[0]) + " " + str(yolo_bbox[1]) + \
                    " " + str(yolo_bbox[2]) + " " + str(yolo_bbox[3]) + "\n")

def visual_json(jsonPath):
    data = json.load(open(jsonPath, "r"))
    # dict_keys(['version', 'flags', 'shapes', 'imagePath', 
    # 'imageData', 'imageHeight', 'imageWidth'])
    print(data.keys())
    print(data['imagePath'])
    # print(data["imageData"])
    print(data["flags"])
    print(data["imageHeight"])
    print(data["imageWidth"])

    # 数据都存在 ”shapes“ 里面, shapes是一个list，里面都是dict, 一个dict是一个框
    print(data["shapes"])

    # dict_keys(['label', 'points', 'group_id', 'shape_type', 'flags'])
    print(data["shapes"][0].keys())
    print(len(data["shapes"]))
    print(data["shapes"][0]["label"])

if __name__ == "__main__":
    json_folder_path = "/home/marco/Estudiar/DeepLearning/Project/yolo_v3_coco/datas/simpleDatasets/labelme"
    json_names = os.listdir(json_folder_path)
    for item in json_names:
        print("============> ", item)
        if not item == "txt":
            decode_json(json_folder_path, item)
        
    # visual_json(os.path.join(json_folder_path, json_names[0]))