import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np

import onnx
#import caffe2.python.onnx.backend as onnx_caffe2

import geffnet


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--output', metavar='ONNX_FILE',default='./pth_models/20211120-214118-resnet50-32/model_best-resnet-50.onnx',
                    help='output model filename')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: dpn92)')
parser.add_argument('--img-size', default=32, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--num-classes', type=int, default=10,
                    help='Number classes in dataset')
parser.add_argument('--checkpoint', default='./pth_models/20211120-214118-resnet50-32/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')


def main():
    args = parser.parse_args()

    if not args.checkpoint:
        args.pretrained = True
    else:
        args.pretrained = False

    # create model
    geffnet.config.set_exportable(True)
    print("==> Creating PyTorch {} model".format(args.model))
    model = geffnet.create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    model.eval()

    # x = torch.randn((1, 3, args.img_size or 224, args.img_size or 224), requires_grad=True)
    x = torch.randn((1, 3, args.img_size or 224, args.img_size or 224), requires_grad=False)
    # x = torch.ones((1, 3, 224, 224), requires_grad=False)
    out = model(x)  # run model once before export trace

    print("==> Exporting model to ONNX format at '{}'".format(args.output))
    input_names = ["input0"]
    output_names = ["output0"]
    optional_args = dict(keep_initializers_as_inputs=True)  # pytorch 1.3 needs this for export to succeed
    try:
        torch_out = torch.onnx._export(
            model, x, args.output, export_params=True, verbose=False,
            input_names=input_names, output_names=output_names, **optional_args)
    except TypeError:
        # fallback to no keep_initializers arg for pytorch < 1.3
        torch_out = torch.onnx._export(
            model, x, args.output, export_params=True, verbose=False,
            input_names=input_names, output_names=output_names)

    print("==> Loading and checking exported model from '{}'".format(args.output))
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)  # assuming throw on error
    print("==> Passed")

    # print("==> Loading model into Caffe2 backend and comparing forward pass.".format(args.output))
    # caffe2_backend = onnx_caffe2.prepare(onnx_model)
    # B = {onnx_model.graph.input[0].name: x.data.numpy()}
    # c2_out = caffe2_backend.run(B)[0]
    # np.testing.assert_almost_equal(torch_out.data.numpy(), c2_out, decimal=5)
    # print("==> Passed")


if __name__ == '__main__':
    main()
