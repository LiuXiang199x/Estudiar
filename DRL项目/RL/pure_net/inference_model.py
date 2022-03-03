import cv2
import time
import numpy as np
from random import randint
from rknn.api import RKNN

def decode( heatmap, scale, offset, landmark, size, threshold=0.1,landmarks = False):
    heatmap = np.squeeze(heatmap)
    scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
    offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
    c0, c1 = np.where(heatmap > threshold)
    if landmarks:
        boxes, lms = [], []
    else:
        boxes = []
    if len(c0) > 0:
        for i in range(len(c0)):
            s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
            o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]
            x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
            x1, y1 = min(x1, size[1]), min(y1, size[0])
            boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
            if landmarks:
                lm = []
                for j in range(5):
                    lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                    lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                lms.append(lm)
        boxes = np.asarray(boxes, dtype=np.float32)
        keep = nms(boxes[:, :4], boxes[:, 4], 0.3)
        boxes = boxes[keep, :]
        if landmarks:
            lms = np.asarray(lms, dtype=np.float32)
            lms = lms[keep, :]
    if landmarks:
        return boxes, lms
    else:
        return boxes

def nms(boxes, scores, nms_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=np.bool)

    keep = []
    for _i in range(num_detections):
        i = order[_i]
        if suppressed[i]:
            continue
        keep.append(i)

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_detections):
            j = order[_j]
            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh:
                suppressed[j] = True

    return keep 
       


def main():

    input_size = (1920,1088)
    decode_input_size = (1088,1920)
   
    
    #导入模型
    rknn = RKNN()
    #非量化模型
    #rknn.load_rknn('./centerface_1088_1920.rknn')
    #量化模型
    rknn.load_rknn('./centerface_quantization_1088_1920.rknn')
    ret = rknn.init_runtime(target='rk1808', target_sub_class='AICS')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    
    #在图片上进行检测
    while True:
        img = cv2.imread('C:\\Users\\Administrator\\Desktop\\1.jpg')
        #模型需要的宽度有所变化
        img = cv2.resize(img,(1920,1088))
        frame = img.copy()
        #增加一个维度
        frame = frame[:, :, :, np.newaxis]
        #转换为模型需要的输入维度(1, 3, 1088, 1920)
        frame = frame.transpose([3, 2, 0, 1])
        print(frame.shape)
        
        t = time.time()
        output = rknn.inference(inputs=[frame], data_format="nchw")
        print("time:", time.time()-t)
        
        heatmap = output[0].reshape(1, 1, 272, 480)
        scale   = output[1].reshape(1, 2, 272, 480)
        offset  = output[2].reshape(1, 2, 272, 480)
        lms     = output[3].reshape(1, 10,272, 480)
        
        dets , lms = decode(heatmap, scale, offset, lms, decode_input_size, threshold=0.58,landmarks = True)
        print("检测到人脸数%d" %(len(dets),))
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(img, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        for lm in lms:
            cv2.circle(img, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)
            cv2.circle(img, (int(lm[2]), int(lm[3])), 2, (0, 0, 255), -1)
            cv2.circle(img, (int(lm[4]), int(lm[5])), 2, (0, 0, 255), -1)
            cv2.circle(img, (int(lm[6]), int(lm[7])), 2, (0, 0, 255), -1)
            cv2.circle(img, (int(lm[8]), int(lm[9])), 2, (0, 0, 255), -1)
        cv2.imshow('out',img)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    rknn.release()

if __name__ == "__main__":
    main()
