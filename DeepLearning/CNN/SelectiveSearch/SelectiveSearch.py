import skimage.data
from skimage.segmentation import clear_border
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#1. 用户生成原始区域集的函数，其中用到了felzenszwalb图像分割算法。每一个区域都有一个编号，将编号并入图片中，方便后面的操作
def _generate_segments(im_orig, scale, sigma, min_size):
    """
        segment smallest regions by the algorithm of Felzenswalb and
        Huttenlocher
    """

    # open the Image
    #计算Felsenszwalb的基于有效图的图像分割。
    im_mask = skimage.segmentation.felzenszwalb(skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,min_size=min_size)
    #im_mask对每一个像素都进行编号

    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask

    return im_orig

#2. 计算两个区域的相似度
# 论文中考虑了四种相似度 -- 颜色，纹理，尺寸，以及交叠。
#
# 其中颜色和纹理相似度，通过获取两个区域的直方图的交集，来判断相似度。
#
# 最后的相似度是四种相似度的加和
def _sim_colour(r1, r2):
    """
        calculate the sum of histogram intersection of colour
    """
    # zip()函数是打包函数
    """
    a = [1,2,3]
    b = [4,5,6]
    c = [4,5,6,7,8]
    zipped = zip(a,b)     # 打包为元组的列表
    [(1, 4), (2, 5), (3, 6)]
    zip(a,c)              # 元素个数与最短的列表一致
    [(1, 4), (2, 5), (3, 6)]
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
        calculate the sum of histogram intersection of texture
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    """
        calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
            (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
            * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    # 计算两个类别的相似度
    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


# 3. 用于计算颜色和纹理的直方图的函数
#
# 颜色直方图：将色彩空间转为HSV，每个通道下以bins=25计算直方图，这样每个区域的颜色直方图有25*3=75个区间。 对直方图除以区域尺寸做归一化后使用下式计算相似度：
#
#
#
# 纹理相似度：论文采用方差为1的高斯分布在8个方向做梯度统计，然后将统计结果（尺寸与区域大小一致）以bins=10计算直方图。直方图区间数为8*3*10=240（使用RGB色彩空间）。这里是用了LBP（local binary pattern）获取纹理特征，建立直方图，其余相同
#
#
#
# 其中，是直方图中第个bin的值。

def _calc_colour_hist(img):
    # 输入的img参数是每一类别的所有像素点的hsv值
    """
        calculate colour histogram for each region  为每个区域计算颜色直方图
        the size of output histogram will be BINS * COLOUR_CHANNELS(3)
        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
        extract HSV
    """

    BINS = 25
    hist = numpy.array([])
    for colour_channel in (0, 1, 2):
        # extracting one colour channel
        # 将输入的参数img各个像素带的第1，2，3hsv色道值提取出来，所以c数组是一维的，c的长度和img是相同的
        c = img[:, colour_channel]
        # calculate histogram for each colour and join to the result
        # numpy.concatenate是拼接函数，将两个函数拼接起来
        # numpy.histogram是计算数据的直方图，即统计哪个数据段中有多少数据，第一个参数是数据矩阵，第二个参数是每个数据段的差距，这里定义成了25，第三个参数是统计的最大最小值
        # 这一步就是将这个类别的三个色道的直方统计拼接在一起
        hist = numpy.concatenate([hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1 normalize
    hist = hist / len(img)
    return hist


def _calc_texture_gradient(img):
    """
        calculate texture gradient for entire image
        The original SelectiveSearch algorithm proposed Gaussian derivative
        for 8 orientations, but we use LBP instead.
        output will be [height(*)][width(*)]
    """

    ret = numpy.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(img[:, :, colour_channel], 8, 1.0)


    return ret


def _calc_texture_hist(img):
    """
        calculate texture histogram for each region
        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10

    hist = numpy.array([])

    for colour_channel in (0, 1, 2):
        # mask by the colour channel
        fd = img[:, colour_channel]

        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = numpy.concatenate([hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])

    # L1 Normalize
    hist = hist / len(img)

    return hist


#4. 提取区域的尺寸，颜色和纹理特征
def _extract_regions(img):
    R = {}

    # get hsv image
    # 每个像素点都是三个小于1的HSV
    hsv = skimage.color.rgb2hsv(img[:, :, :3])

    # pass 1: count pixel positions
    # 遍历图片像素块,将每个类别最大以及最小的x、y坐标记录在字典R中
    for y, i in enumerate(img):

        for x, (r, g, b, l) in enumerate(i):

            # initialize a new region
            if l not in R:
                R[l] = {"min_x": 0xffff, "min_y": 0xffff,"max_x": 0, "max_y": 0, "labels": [l]}

            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # pass 2: calculate texture gradient
    tex_grad = _calc_texture_gradient(img)



    # pass 3: calculate colour histogram of each region
    #k是种类，v是此类别的minx,maxx,miny,maxy
    for k, v in list(R.items()):
        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]# 将输入第k类别的像素的hsv提取出来

        R[k]["size"] = len(masked_pixels / 4)#记录该类别的像素总数（很迷，不知道为啥除以4，除不除4结果都一样）
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)#记录该类比的hsv三道的分布直方统计

        # texture histogram
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])#与上一步类似


    #这里返回的R记录了该图像每个类别的信息：mix_x,min_y,max_x,max_y,size,hist_c,hist_t
    return R


#5. 找邻居 -- 通过计算每个区域与其余的所有区域是否有相交，来判断是不是邻居
# 参数regions：R记录了该图像每个类别的信息：mix_x,min_y,max_x,max_y,size,hist_c,hist_t
def _extract_neighbours(regions):
    def intersect(a, b):# a和b都是两个类别
        #b的最小x在a的最小x和最大x之间并且b的最小y在a的最小y和最大y之间  或者
        #b的最大x在a的最小x和最大x之间并且b的最大y在a的最小y和最大y之间  或者
        #b的最小x在a的最小x和最大x之间并且b的最大y在a的最小y和最大y之间  或者
        #b的最大x在a的最小x和最大x之间并且b的最小y在a的最小y和最大y之间
        #总而言之，大概就是b的有一部分在a的里面
        if (a["min_x"] < b["min_x"] < a["max_x"]and a["min_y"] < b["min_y"] < a["max_y"]) \
            or (a["min_x"] < b["max_x"] < a["max_x"]and a["min_y"] < b["max_y"] < a["max_y"]) \
            or (a["min_x"] < b["min_x"] < a["max_x"]and a["min_y"] < b["max_y"] < a["max_y"]) \
            or (a["min_x"] < b["max_x"] < a["max_x"]and a["min_y"] < b["min_y"] < a["max_y"]):
                return True
        return False

    R = list(regions.items()) # items()取regions的每个元素，即每个类别的信息
    neighbours = []
    for cur, a in enumerate(R[:-1]):# enumerate（）用于迭代，可返回下标和值，即cur是下标，从0开始 ,R[:-1]表示最后一个不遍历
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):#拿当前类别a与a之后的所有类别进行比较
                neighbours.append((a, b))#将是邻居的两个类装进数组中

    return neighbours

# 6. 合并两个区域的函数
# 参数是两个区域的信息，将两个区域合并成一个区域
def _merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt

# 7. 主函数 -- Selective Search
#
# scale：图像分割的集群程度。值越大，意味集群程度越高，分割的越少，获得子区域越大。默认为1
#
# signa: 图像分割前，会先对原图像进行高斯滤波去噪，sigma即为高斯核的大小。默认为0.8
#
# min_size  : 最小的区域像素点个数。当小于此值时，图像分割的计算就停止，默认为20
#
# 每次选出相似度最高的一组区域（如编号为100和120的区域），进行合并，得到新的区域（如编号为300）。
# 后计算新的区域300与区域100的所有邻居和区域120的所有邻居的相似度，加入区域集S。不断循环，知道S为空，
# 此时最后只剩然下一个区域，而且它的像素数会非常大，接近原始图片的像素数，因此无法继续合并。最后退出程序。


def selective_search(im_orig, scale=1.0, sigma=0.8, min_size=50):
    '''Selective Search
    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''
    #当图片不是三通道时，引发异常
    assert im_orig.shape[2] == 3, "3ch image is expected"

    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)]
    img = _generate_segments(im_orig, scale, sigma, min_size)


    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img)   #R记录了该图像每个类别的信息：mix_x,min_y,max_x,max_y,size,hist_c,hist_t

    # extract neighbouring information
    neighbours = _extract_neighbours(R)

    # calculate initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        # ai，bi指的是两个类别的编号，ar，br指的是每个类别的信息
        #S是一个字典，key是两个类别的编号，value是这两个类别的相似度
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    # hierarchal search
    while S != {}:

        # get highest similarity
        # 找到S中相似度最高的两个类别编号i,j
        # sorted函数对S进行排序，[-1][0]指的是取出排序后最后一项（value即相似度最大）的第一个元素（即两个类别的编号）
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # merge corresponding regions
        #key（）函数，返回字典所有的键
        t = max(R.keys()) + 1.0#意思是新建一个新的区域编号
        R[t] = _merge_regions(R[i], R[j])

        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                #将S中有类别i和类别j的项目找出来放入key_to_delete数组中
                key_to_delete.append(k)

        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]

        # calculate similarity set with the new region
        # 计算新区域和其他区域的相似度
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in list(R.items()):
        #k是编号，r存储了该类别的信息
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions




# 读入一张skimage自带的一幅图片，该图片大小是512*512*3的
# img = skimage.data.astronaut()
# img_lbl, regions = selective_search(img, scale=500, sigma=0.9, min_size=10)
# print(regions)


def main():
    # 加载图片数据
    img = skimage.data.astronaut()

    '''
    执行selective search，regions格式如下
    [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
    ]
    '''
    img_lbl, regions = selective_search(img, scale=500, sigma=0.9, min_size=10)
    print(img_lbl.shape)
    # 计算一共分割了多少个原始候选区域
    temp = set() # set() 函数创建一个无序不重复元素集
    for i in range(img_lbl.shape[0]):
        for j in range(img_lbl.shape[1]):
            # temp存储了所有的类别编号
            temp.add(img_lbl[i, j, 3])

    print(len(temp))  # 286

    # 计算利用Selective Search算法得到了多少个候选区域
    print(len(regions))  # 570
    # 创建一个集合 元素不会重复，每一个元素都是一个list(左上角x，左上角y,宽,高)，表示一个候选区域的边框
    candidates = set()
    for r in regions:
        # 排除重复的候选区
        if r['rect'] in candidates:
            continue
        # 排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)
        if r['size'] < 2000:
            continue
        # 排除扭曲的候选区域边框  即只保留近似正方形的
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # 在原始图像上绘制候选区域边框
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

main()
