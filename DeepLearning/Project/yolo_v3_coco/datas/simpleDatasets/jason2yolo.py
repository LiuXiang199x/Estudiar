from cProfile import label
from importlib import import_module
import json
import os

name2id = {"person": 0, "cloud": 1}

def decode_json(json_folder_path, json_name):

    txt_name = "/home/marco/Estudiar/DeepLearning/Project/yolo_v3_coco/datas/simpleDatasets/labelme/txt/" + json_name[0] + ".txt"
    txt_file = open(txt_name, "w")

    json_path = os.path.join(json_folder_path, json_name)
    data = json.load(open(json_path, "r", encoding="gb2312"))

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    for i in data["shapes"]:
        label_name = i['label']
        if (i['shape_type'] == "rectangle"):
            x1 = int(i['points'][0][0])

if __name__ == "__main__":
    json_folder_path = "/home/marco/Estudiar/DeepLearning/Project/yolo_v3_coco/datas/simpleDatasets/labelme"
    json_names = os.listdir(json_folder_path)
    for item in json_names:
        print("============> ", item)
        decode_json(json_folder_path, item)
        break

