#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys

def ReadImage(png_name):
    img = cv2.imread(png_name)
    res = []
    for x in range(len(img)):
        row = []
        for y in range(len(img[x])):
            label = 0
            for i in range(len(img[x][y])):
                label = label * 1000 + img[x][y][i]
            row.append(label)
        res.append(row)
    return res

def Compare(gt, rs):
    assert len(gt) == len(rs) and len(gt[0]) == len(rs[0])

    overlap = dict()

    gt_labels = dict()
    rs_labels = dict()
    for x in range(len(gt)):
        for y in range(len(gt[0])):
            label_a = gt[x][y]
            label_b = rs[x][y]

            if label_a != 0 and label_a != 128138135: 
                if label_a not in gt_labels:
                    gt_labels[label_a] = 0
                gt_labels[label_a] += 1 

            if label_b != 0:
                if label_b not in rs_labels:
                    rs_labels[label_b] = 0
                rs_labels[label_b] += 1

            if label_a != 0 and label_a != 128138135 and label_b != 0:
                if label_a not in overlap:
                    overlap[label_a] = dict()
                    overlap[label_a][label_b] = 0
                elif label_b not in overlap[label_a]:
                    overlap[label_a][label_b] = 0

                overlap[label_a][label_b] += 1

    for gt_label in gt_labels:
        for rs_label in rs_labels:
            if gt_label not in overlap:
                overlap[gt_label] = dict()
                overlap[gt_label][rs_label] = 0
            elif rs_label not in overlap[gt_label]:
                overlap[gt_label][rs_label] = 0

    recall_micro = 0
    recall_macro = 0
    gt_total = 0
    for gt_label in gt_labels:
        max_overlap = 0
        for rs_label in rs_labels:
            if max_overlap < overlap[gt_label][rs_label]:
                max_overlap = overlap[gt_label][rs_label]
        recall_micro += 1.0 * max_overlap / gt_labels[gt_label]
        recall_macro += 1.0 * max_overlap 
        gt_total += gt_labels[gt_label]

    recall_micro /= len(gt_labels)
    recall_macro /= gt_total

    #print('recall_micro', recall_micro)
    #print('recall_macro', recall_macro)
    print('recall:', recall_micro)

    precision_micro = 0
    precision_macro = 0
    rs_total = 0
    for rs_label in rs_labels:
        max_overlap = 0
        for gt_label in gt_labels:
            if max_overlap < overlap[gt_label][rs_label]:
                max_overlap = overlap[gt_label][rs_label]
        precision_micro += 1.0 * max_overlap / rs_labels[rs_label]
        precision_macro += 1.0 * max_overlap
        rs_total += rs_labels[rs_label]
    precision_micro /= len(rs_labels)
    precision_macro /= rs_total

    #print('precision_micro', precision_micro)
    #print('precision_macro', precision_macro)
    print('precision:', precision_micro)


def main():
    #ground_truth_png = 'iot_id-2269-map_id-1646561988285455674.png'
    #room_seg_png = 'iot_id-2269-map_id-1646561988285455674_room_seg.png'
    assert len(sys.argv) == 3
    ground_truth_png = sys.argv[1]
    room_seg_png = sys.argv[2]
    gt = ReadImage(ground_truth_png)
    rs = ReadImage(room_seg_png)

    Compare(gt, rs)


if __name__ == '__main__':
    main()

