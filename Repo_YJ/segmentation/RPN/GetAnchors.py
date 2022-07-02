import numpy as np


# base_anchor是最开始的点,(0,0,15,15)坐标点左上角和右下角
# ratios=[0.5, 1, 2]是需要变换的长宽比，scales=2 ** np.arange(3, 6))就是面积的比。
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],scales=2 ** np.arange(3, 6)):

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    print("base_anchor: ", base_anchor)

    # 生成经过长宽比变化后的三种框的坐标信息，最后一步anchors 的生成是 按照面积大小在次变化。
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    print("ratio_anchors: ", ratio_anchors)

    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    
    return anchors


def _ratio_enum(anchor, ratios):
    # 获取宽高和中心点坐标
    w,h,x_ctr,y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    print("size_ratios: ", size_ratios)

    # np.round（）去掉小数点，_mkanchors()是给定anchor的中心点和宽高求出anchor的左上点和右下点坐标。 
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)    
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


# _whctrs()是给定anchor左上点和右下点坐标求出anchor的中心点和宽高。x_ctr,y_ctr中心点坐标。
def _whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, np.newaxis] 
    hs = hs[:, np.newaxis]  
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


if __name__ == '__main__':
    a = generate_anchors() 
    print(a)

