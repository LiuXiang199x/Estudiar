import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .anchor_target import anchor_target
from .proposal import proposal
import config as cfg
from ..utils.smooth_L1 import smooth_L1


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        self.anchor_scales = np.array([8, 16, 32])
        self.anchor_ratios = [0.5, 1, 2]
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        self.rpn_conv = nn.Conv2d(1024, 512, 3, 1, 1)
        self.rpn_cls = nn.Conv2d(512, self.num_anchors * 2, 1, 1, 0)
        self.rpn_reg = nn.Conv2d(512, self.num_anchors * 4, 1, 1, 0)

    def forward(self, feature, gt_boxes, im_info):
        batch_size, _, height, width = feature.size()

        rpn_features = F.relu(self.rpn_conv(feature), inplace=True)

        # classification and regression
        rpn_cls_score = self.rpn_cls(rpn_features)
        rpn_reg = self.rpn_reg(rpn_features)

        rpn_cls_score_reshape = rpn_cls_score.view(batch_size, 2, self.num_anchors, height, width)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob = rpn_cls_prob_reshape.view(batch_size, 2 * self.num_anchors, height, width)

        # calculate proposals
        rois = proposal(rpn_cls_prob.data, rpn_reg.data, im_info, self.training)

        rpn_cls_loss = 0
        rpn_reg_loss = 0
        _rpn_train_info = {}
        if self.training:
            # calculate anchor target
            rpn_label_targets, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target(rpn_cls_prob.data, gt_boxes, im_info)

            keep_inds = torch.nonzero(rpn_label_targets != -1).view(-1)
            rpn_label_targets_keep = Variable(rpn_label_targets[keep_inds]).long()

            keep_inds = Variable(keep_inds)

            rpn_cls_score = rpn_cls_score_reshape.permute(0, 3, 4, 2, 1).contiguous().view(-1, 2)

            # cross entropy loss
            rpn_cls_loss = F.cross_entropy(rpn_cls_score[keep_inds, :], rpn_label_targets_keep)

            # smooth L1 loss
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_reg_loss = smooth_L1(rpn_reg, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, dim=[1, 2, 3])

            if cfg.VERBOSE:
                _rpn_fg_inds = torch.nonzero(rpn_label_targets == 1).view(-1)
                _rpn_bg_inds = torch.nonzero(rpn_label_targets == 0).view(-1)
                _rpn_num_fg = _rpn_fg_inds.size(0)
                _rpn_num_bg = _rpn_bg_inds.size(0)
                _rpn_pred_data = rpn_cls_prob_reshape.permute(0, 3, 4, 2, 1).contiguous().view(-1, 2)[:, 1]
                _rpn_pred_info = (_rpn_pred_data >= 0.4).long()
                _rpn_tp = torch.sum(rpn_label_targets[_rpn_fg_inds].long() == _rpn_pred_info[_rpn_fg_inds])
                _rpn_tn = torch.sum(rpn_label_targets[_rpn_bg_inds].long() == _rpn_pred_info[_rpn_bg_inds])
                _rpn_train_info['rpn_num_fg'] = _rpn_num_fg
                _rpn_train_info['rpn_num_bg'] = _rpn_num_bg
                _rpn_train_info['rpn_tp'] = _rpn_tp.item()
                _rpn_train_info['rpn_tn'] = _rpn_tn.item()

        return rois, rpn_cls_loss, rpn_reg_loss, _rpn_train_info

import torch
import numpy as np
import config as cfg
from .generate_anchors import generate_anchors
from ..utils.bbox_operations import bbox_transform_inv, clip_boxes
from ..nms.nms_wrapper import nms


def proposal(rpn_cls_prob, rpn_reg, im_info, train_mode):


    batch_size, _, height, width = rpn_cls_prob.size()

    assert batch_size == 1, 'only support single batch'

    im_info = im_info[0]

    if train_mode:
        pre_nms_top_n = cfg.TRAIN.RPN_PRE_NMS_TOP_N
        post_nms_top_n = cfg.TRAIN.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TRAIN.RPN_NMS_THRESH
        min_size = cfg.TRAIN.RPN_MIN_SIZE
    else:
        pre_nms_top_n = cfg.TEST.RPN_PRE_NMS_TOP_N
        post_nms_top_n = cfg.TEST.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TEST.RPN_NMS_THRESH
        min_size = cfg.TEST.RPN_MIN_SIZE

    anchor_scales = cfg.RPN_ANCHOR_SCALES
    anchor_ratios = cfg.RPN_ANCHOR_RATIOS
    feat_stride = cfg.FEAT_STRIDE

    # generate anchors
    _anchors = generate_anchors(base_size=feat_stride, scales=anchor_scales, ratios=anchor_ratios)
    num_anchors = _anchors.shape[0]

    A = num_anchors
    K = int(height * width)
    shift_x = np.arange(0, width) * cfg.FEAT_STRIDE
    shift_y = np.arange(0, height) * cfg.FEAT_STRIDE
    shifts_x, shifts_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shifts_x.ravel(), shifts_y.ravel(), shifts_x.ravel(), shifts_y.ravel())).transpose()

    all_anchors = _anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
    all_anchors = all_anchors.reshape(-1, 4)
    num_all_anchors = all_anchors.shape[0]

    assert num_all_anchors == K * A

    all_anchors = torch.from_numpy(all_anchors).type_as(rpn_cls_prob)

    rpn_reg = rpn_reg.permute(0, 2, 3, 1).contiguous().view(-1, 4)

    # generate all rois
    proposals = bbox_transform_inv(all_anchors, rpn_reg)
    proposals = clip_boxes(proposals, im_info)

    # filter proposals
    keep_inds = _filter_proposal(proposals, min_size)

    proposals_keep = proposals[keep_inds, :]

    # proposal prob
    proposals_prob = rpn_cls_prob[:, num_anchors:, :, :]
    proposals_prob = proposals_prob.permute(0, 2, 3, 1).contiguous().view(-1)

    proposals_prob_keep = proposals_prob[keep_inds]

    # sort prob
    order = torch.sort(proposals_prob_keep, descending=True)[1]

    top_keep = order[:pre_nms_top_n]

    proposals_keep = proposals_keep[top_keep, :]
    proposals_prob_keep = proposals_prob_keep[top_keep]

    # nms
    keep = nms(torch.cat([proposals_keep, proposals_prob_keep.view(-1, 1)], dim=1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
    keep = keep.long().view(-1)
    top_keep = keep[:post_nms_top_n]

    proposals_keep = proposals_keep[top_keep, :]
    proposals_prob_keep = proposals_prob_keep[top_keep]

    rois = proposals_keep.new_zeros((proposals_keep.size(0), 5))

    rois[:, 1:] = proposals_keep

    return rois


def _filter_proposal(proposals, min_size):

    width = proposals[:, 2] - proposals[:, 0] + 1
    height = proposals[:, 3] - proposals[:, 1] + 1

    keep_inds = ((width >= min_size) & (height >= min_size))

    return keep_inds

