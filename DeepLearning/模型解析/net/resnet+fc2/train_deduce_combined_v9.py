# Usage: python train_deduce_combined_v9.py -a resnet18 -b=30 -j=8 -p=100 --config config/nanodet_irbt_list_dataset.yml --model workspace/nanodet-plus-m-1.5x_416/model_best/nanodet_model_best.pth --num_classes=5

# Anwesan Pal
import argparse
import os
import shutil

import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import resnet
import wideresnet
import pdb
import matplotlib.pyplot as plt
# from yolov3.detect import detect
from config_v2 import places_dir


import time
import cv2
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

os.environ['CUDA_VISIBLE_DEVICES']= "7"

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


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Training of Combined model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=7, type=int, 
                    help='num of class in the model')

#parser.add_argument(
#    "demo", default="list", help="demo type, eg. image, video and webcam and list"
#)
parser.add_argument("--config", help="model config file path")
parser.add_argument("--model", help="model file path")
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


best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.lower().startswith('wideresnet'):
        # a customized resnet model with last feature map size as 14x14 for better class activation mapping
        model = wideresnet.resnet50(num_classes=args.num_classes)
    else:
        # model = models.__dict__[args.arch](num_classes=args.num_classes)
        model = resnet.resnet18(num_classes=args.num_classes)

    if args.arch.lower().startswith('alexnet') or args.arch.lower().startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # load the pre-trained weights from open source
    model_file = 'models/resnet18_best_home.pth.tar'
    pre_checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    #pre_state_dict = {str.replace(k, 'module.', ''): v for k, v in pre_checkpoint['state_dict'].items()}
    pre_state_dict = pre_checkpoint['state_dict']
    del pre_state_dict['module.fc.weight']
    del pre_state_dict['module.fc.bias']
    model_dict = model.state_dict()
    model_dict.update(pre_state_dict)
    model.load_state_dict(model_dict)


    print(model)

    class Object_Linear(nn.Module):
        def __init__(self):
            super(Object_Linear, self).__init__()
            self.fc = nn.Linear(13, 512)

        def forward(self, x):
            out = self.fc(x)
            return out
    object_idt = Object_Linear()
    object_idt = torch.nn.DataParallel(object_idt).cuda()

    class LinClassifier(nn.Module):
        def __init__(self,num_classes):
            super(LinClassifier, self).__init__()
            self.num_classes = num_classes
            self.fc = nn.Linear(1024, num_classes)

        def forward(self, conv, idt):
            out = torch.cat((conv,idt),1)
            out = self.fc(out)
            return out
    classifier = LinClassifier(args.num_classes)
    classifier = torch.nn.DataParallel(classifier).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model_state_dict'])
            object_idt.load_state_dict(checkpoint['obj_state_dict'])
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        for param in model.parameters():
            param.requires_grad = False

    cudnn.benchmark = True

    # Data loading code
    data_dir = places_dir + '/data'
    traindir = os.path.join(data_dir, 'train')
    print(traindir)
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        """

        # override the __getitem__ method. this is the method dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path

    train_dataset = ImageFolderWithPaths(traindir, transforms.Compose([
            #transforms.RandomSizedCrop(224),
            transforms.Resize((360, 360)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = ImageFolderWithPaths(valdir, transforms.Compose([
            #transforms.Scale(256),
            #transforms.CenterCrop(224),
            transforms.Resize((360, 360)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    params = list(object_idt.parameters())+list(classifier.parameters())
    # params = list(model.parameters())

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    local_rank = 0
    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device="cuda:0")

    if args.evaluate:
        validate(val_loader, model, object_idt, classifier, criterion, predictor)
        return
    
    accuracies_list = []


    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, object_idt, classifier, criterion, optimizer, epoch, predictor)

        # evaluate on validation set
        prec1 = validate(val_loader, model, object_idt, classifier, criterion, predictor)

        accuracies_list.append("%.2f"%prec1.tolist())
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'model_state_dict': model.state_dict(),
            'obj_state_dict': object_idt.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.arch.lower())
    print("The best accuracy obtained during training is = {}".format(best_prec1))


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
    # print("=====", all_box)
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
    # print("=====", one_hot_selected_objects)
    return one_hot_selected_objects


def train(train_loader, model, object_idt, classifier, criterion, optimizer, epoch, predictor):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    object_idt.train()
    classifier.train()

    end = time.time()
    for i, (input, target, path) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output_conv = model(input_var)
        obj_id_batch = []
        for j in range(len(path)):
            # objects, class_names = detect(args.cfg, args.weight, path[j],args.namesfile)
            # obj_hot_vector = get_hot_vector(objects, class_names)

            #print("=====", path[j])
            meta, res = predictor.inference(path[j])
            obj_hot_vector = get_hot_vector_from_box(res[0], 0.35)

            obj_id_batch.append(obj_hot_vector)
        t = torch.autograd.Variable(torch.FloatTensor(obj_id_batch))
        output_idt = object_idt(t)
        output = classifier(output_conv,output_idt)  
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, object_idt, classifier, criterion, predictor):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    object_idt.eval()
    classifier.eval()

    end = time.time()
    for i, (input, target, path) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():

            # compute output
            output_conv = model(input)
            obj_id_batch = []
            for j in range(len(path)):
                #objects, class_names = detect(args.cfg, args.weight, path[j],args.namesfile)
                #obj_hot_vector = get_hot_vector(objects, class_names)
                #print("=====", path[j])
                meta, res = predictor.inference(path[j])
                obj_hot_vector = get_hot_vector_from_box(res[0], 0.35)

                obj_id_batch.append(obj_hot_vector)
            t = torch.autograd.Variable(torch.FloatTensor(obj_id_batch))
            output_idt = object_idt(t)
            output = classifier(output_conv,output_idt)  
            loss = criterion(output_conv, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss, input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest_combined_v9.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest_combined_v9.pth.tar', filename + '_best_combined_v9.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
