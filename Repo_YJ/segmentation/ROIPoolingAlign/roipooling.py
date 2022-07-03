# -*- coding:UTF-8 -*-
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchv
 
 
def roi_pooling(input, rois, size=(7, 7), spatial_scale=1.0):
	assert rois.dim() == 2
	assert rois.size(1) == 5
	output = []
	rois = rois.data.float()
	num_rois = rois.size(0)
 
	rois[:, 1:].mul_(spatial_scale)
	rois = rois.long()
	for i in range(num_rois):
		roi = rois[i]
		im_idx = roi[0]
		im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
		output.append(F.adaptive_max_pool2d(im, size))
 
	output = torch.cat(output, 0)
	if has_backward:
		#        output.backward(output.data.clone())
		output.sum().backward()
	return output
 
# def relation_pooling(input, roisA,roisB, size=(7, 7), spatial_scale=1.0):  # pytorch version use for loop !!!
#
# 	output = []
# 	roisA = roisA.data.float()
# 	roisB = roisB.data.float()
# 	num_rois = roisA.size(0)
#
# 	h = input.size(2)
# 	w = input.size(3)
#
# 	roisA[:, 1:].mul_(spatial_scale)
# 	roisB[:, 1:].mul_(spatial_scale)
# 	roisA = roisA.long()
# 	roisB = roisB.long()
#
#
# 	for i in range(num_rois):
# 		roiA = roisA[i]
# 		roiB = roisB[i]
# 		xmin = min(roiA[1], roiB[1])
# 		ymin = min(roiA[2], roiB[2])
# 		xmax = max(roiA[3], roiB[3])
# 		ymax = max(roiA[4], roiB[4])
# 		mask = Variable(torch.zeros(h,w)).cuda()
# 		mask[roiA[2]:(roiA[4] + 1), roiA[1]:(roiA[3] + 1)] = 1
# 		mask[roiB[2]:(roiB[4] + 1), roiB[1]:(roiB[3] + 1)] = 1
# 		input = input * mask.unsqueeze(0).unsqueeze(1)
# 		unionbox = torch.LongTensor([0,xmin,ymin,xmax,ymax])
# 		im = input[..., unionbox[2]:(unionbox[4] + 1), unionbox[1]:(unionbox[3] + 1)]
# 		output.append(F.adaptive_max_pool2d(im, size))
#
# 	output = torch.stack(output, dim=0).squeeze()
#
# 	if has_backward:
# 		#        output.backward(output.data.clone())
# 		output.sum().backward()
# 	return output
 
def create_rois(config):
 
	rois = torch.rand((config[2], 5))
	rois[:, 0] = rois[:, 0] * config[0]
	rois[:, 1:] = rois[:, 1:] * config[1]
	for j in range(config[2]):
		max_, min_ = max(rois[j, 1], rois[j, 3]), min(rois[j, 1], rois[j, 3])
		rois[j, 1], rois[j, 3] = min_, max_
		max_, min_ = max(rois[j, 2], rois[j, 4]), min(rois[j, 2], rois[j, 4])
		rois[j, 2], rois[j, 4] = min_, max_
	rois = torch.floor(rois)
	rois = Variable(rois, requires_grad=False)
	return rois
 
 
if __name__ == '__main__':
	# batch_size, img_size, num_rois
	config = [1, 50, 300]
	T = 50
	has_backward = True
 
	start = time.time()
	x = Variable(torch.rand((config[0], 512, config[1], config[1])),requires_grad=True).cuda()
	rois = create_rois(config).cuda()
	for t in range(T):
		output = roi_pooling(x,rois)
	print('time: {}, batch_size: {}, size: {}, num_rois: {}'.format((time.time() - start) / T,
		                                                                config[0],
		                                                                config[1],
		                                                                config[2]))
 
 
	# start = time.time()
	# x = Variable(torch.rand((config[0], 512, config[1], config[1])),requires_grad=True).cuda()
	# roisA = create_rois(config).cuda()
	# roisB = create_rois(config).cuda()
	# for t in range(T):
	# 	output = relation_pooling(x, roisA, roisB)
	# print('time: {}, batch_size: {}, size: {}, num_rois: {}'.format((time.time() - start) / T,
	# 	                                                                config[0],
	# 	                                                                config[1],
	# 	                                                                config[2]))
 