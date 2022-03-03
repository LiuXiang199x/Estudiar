# python test_combined_v8.py --dataset=data --envtype=home --config config/nanodet_irbt_list_dataset.yml --model workspace/nanodet-plus-m-1.5x_416/model_best/nanodet_model_best.pth


# zhuhao: For predicting objects

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import resnet
# from yolov3.detect import detect
from config_v2 import places_dir,sun_dir,vpc_dir,home_data_dir
# from train_deduce_combined import get_hot_vector

import time
import cv2
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

import xlwt


def get_hot_vector_from_box(dets, score_thresh):
    all_box = []
    one_hot_all_objects = [0] * 28
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
                one_hot_all_objects[label] = 1
    all_box.sort(key=lambda v: v[5])
    #print("=====", all_box)
    one_hot_selected_objects = [0] * 13
    one_hot_selected_objects[0] = one_hot_all_objects[0]
    one_hot_selected_objects[1] = one_hot_all_objects[2]
    one_hot_selected_objects[2] = one_hot_all_objects[4]
    one_hot_selected_objects[3] = one_hot_all_objects[5]
    one_hot_selected_objects[4] = one_hot_all_objects[6]
    one_hot_selected_objects[5] = one_hot_all_objects[9]
    one_hot_selected_objects[6] = one_hot_all_objects[11]
    one_hot_selected_objects[7] = one_hot_all_objects[15]
    one_hot_selected_objects[8] = one_hot_all_objects[16]
    one_hot_selected_objects[9] = one_hot_all_objects[17]
    one_hot_selected_objects[10] = one_hot_all_objects[18]
    one_hot_selected_objects[11] = one_hot_all_objects[21]
    one_hot_selected_objects[12] = one_hot_all_objects[23]
    #print("=====", one_hot_selected_objects)
    return one_hot_selected_objects


parser = argparse.ArgumentParser(description='DEDUCE Combined Evaluation')

parser.add_argument('--dataset',default='places',help='dataset to test')
parser.add_argument('--hometype',default='home1',help='home type to test')
parser.add_argument('--floortype',default='data_0',help='data type to test')
parser.add_argument('--envtype',default='home',help='home or office type environment')


parser.add_argument("--config", help="model config file path")
parser.add_argument("--model", help="model file path")
#parser.add_argument("--path", default="./demo", help="path to images or video")
parser.add_argument("--class_names", default="./input/s1_28_class.txt", help="webcam demo camera id")
parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
parser.add_argument(
    "--save_result",
    default=False,
    help="whether to save the inference result of input/vis",
)
parser.add_argument("--save_dir", default="./predict_imgs", help="path to images or video")
parser.add_argument("--results_dir", default="./input/detection-results-640/", help="path to images or video")
parser.add_argument("--ground_dir", default="./input/ground-truth-640/", help="path to images or video")
parser.add_argument("--out_txt", default="output.txt", help="path to images or video")


global args
args = parser.parse_args()

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_best_combined_v8.pth.tar' % arch

class Object_Linear(nn.Module):
    def __init__(self):
        super(Object_Linear, self).__init__()
        self.fc = nn.Linear(13, 512)

    def forward(self, x):
        out = self.fc(x)
        return out
object_idt = Object_Linear()

class LinClassifier(nn.Module):
    def __init__(self,num_classes):
        super(LinClassifier, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, conv, idt):
        out = torch.cat((conv,idt),1)
        out = self.fc(out)
        return out
classifier = LinClassifier(5)

# model = models.__dict__[arch](num_classes=7)
model = resnet.resnet18(num_classes = 5)
#'''
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
#print(checkpoint)
model_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['model_state_dict'].items()}
obj_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['obj_state_dict'].items()}
classifier_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['classifier_state_dict'].items()}
model.load_state_dict(model_state_dict)
object_idt.load_state_dict(obj_state_dict)
classifier.load_state_dict(classifier_state_dict)
print("====== weight loaded")
#'''

model.eval()
object_idt.eval()
classifier.eval()

# load the image transformer
centre_crop = trn.Compose([
        #trn.Resize((256,256)),
        #trn.CenterCrop(224),
        trn.Resize((640, 360)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_data_{}.txt'.format(args.envtype)
# if not os.access(file_name, os.W_OK):
#     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
#     os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

if(args.dataset == 'places'):
    data_dir = places_dir + '/places365_standard_{}'.format(args.envtype)
    valdir = os.path.join(data_dir, 'val')
elif(args.dataset == 'sun'):
    data_dir = sun_dir
    valdir = os.path.join(data_dir, 'test')
elif(args.dataset == 'vpc'):
    data_dir = vpc_dir
    home_dir = os.path.join(data_dir, 'data_'+args.hometype)
    valdir = os.path.join(home_dir,args.floortype)
elif(args.dataset == 'data'):
    data_dir = home_data_dir
    valdir = os.path.join(data_dir, 'val')
    traindir = os.path.join(data_dir, 'train')


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        detect_model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        print(model_path)
        load_model_weight(detect_model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            detect_model = repvgg_det_model_convert(detect_model, deploy_model)
        self.model = detect_model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img


local_rank = 0
load_config(cfg, args.config)
logger = Logger(local_rank, use_tensorboard=False)
predictor = Predictor(cfg, args.model, logger, device="cuda:0")


accuracies_list = []
predicted_count = torch.zeros(5, 3) # 3: correct, predict, ground
detailed_count = torch.zeros(5, 5)

txt_path = "detailed_prediction.txt"
file_txt = open(txt_path, "w")

workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
xls_line = 0

for class_name in os.listdir(valdir):
    correct,count=0,0

    if class_name == 'bed_room':
        class_id = 0
    elif class_name == 'dining_room':
        class_id = 1
    elif class_name == 'drawing_room':
        class_id = 2
    elif class_name == 'others':
        class_id = 3
    elif class_name == 'toilet_room':
        class_id = 4

    for img_name in os.listdir(os.path.join(valdir,class_name)):
        img_dir = os.path.join(valdir,class_name,img_name)
        img = Image.open(img_dir)
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass
        output_conv = model.forward(input_img)
        #objects, class_names = detect(args.cfg, args.weight, img_dir,args.namesfile)
        #obj_hot_vector = get_hot_vector(objects, class_names)
        #print("=====", img_dir)
        meta, res = predictor.inference(img_dir)
        obj_hot_vector = get_hot_vector_from_box(res[0], 0.35)
        t = torch.autograd.Variable(torch.FloatTensor(obj_hot_vector))
        output_idt = object_idt(t)
        output_idt = output_idt.unsqueeze(0)
        logit = classifier(output_conv,output_idt)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        input_txtname = class_name + "/" + img_name + " " + str(int(idx[0])) + "\n"
        file_txt.write(input_txtname)

        worksheet.write(xls_line, 0, class_name + "/" + img_name)
        #worksheet.write(xls_line, 4, "=HYPERLINK(" + "\"" + valdir + "/" + class_name + "/" + img_name + "\"" + ")")
        worksheet.write(xls_line, 1, str(int(idx[0])))
        worksheet.write(xls_line, 2, valdir + "/" + class_name + "/" + img_name)
        worksheet.write(xls_line, 3, str(h_x)[7:-1])

        if classes[idx[0]] == class_name:
            correct += 1
            predicted_count[idx[0], 0] += 1
        if classes[idx[0]] == 'bed_room':
            predicted_count[0, 1] += 1
        elif classes[idx[0]] == 'dining_room':
            predicted_count[1, 1] += 1
        elif classes[idx[0]] == 'drawing_room':
            predicted_count[2, 1] += 1
        elif classes[idx[0]] == 'others':
            predicted_count[3, 1] += 1
        elif classes[idx[0]] == 'toilet_room':
            predicted_count[4, 1] += 1

        predicted_count[class_id, 2] += 1
        detailed_count[class_id, idx[0]] += 1
        xls_line += 1

        count+=1
    accuracy = 100*correct/float(count)
    print('Accuracy of {} class is {:2.2f}%'.format(class_name,accuracy))
    accuracies_list.append(accuracy)
# print('Average test accuracy is = {:2.2f}%'.format(sum(accuracies_list)/len(accuracies_list)))

file_txt.close()
workbook.save('detailed_prediction_logit_v8.xls')

print(predicted_count)
recall = predicted_count[:, 0]/predicted_count[:, 2]
print(recall)
recall_avg = sum(predicted_count[:, 0])/sum(predicted_count[:, 2])
print(recall_avg)

precision = predicted_count[:, 0]/predicted_count[:, 1]
print(precision)
precision_avg = sum(predicted_count[:, 0]/predicted_count[:, 1] * predicted_count[:, 2]/sum(predicted_count[:, 2]))
print(precision_avg)
print(detailed_count)

class_count = predicted_count[:, 2].expand(5, 5).t()
detailed_ratio = detailed_count/class_count
print(detailed_ratio)
print("done")

