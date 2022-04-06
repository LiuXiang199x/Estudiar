目标检测：
    1. 单目标检测很简单：（c_是否检测到目标, x1,y1,x2,y2, cls）损失函数搞好由这三部分组成。

yolo系列（多目标检测）：
    1. 大目标-13*13；中目标-26*26；小目标-52*52。
    2. 算损失函数一样的，只是单目标是一个1*6的vector，而yolo是一个多目标N*6的matrix.
    3. 单目标用的是左上角左边和右下角坐标。yolo是中心点和w，h。
    （中心点和 wh互不影响，中心点确定框的位置，但是不确定框的大小和形状，w和h来确定框的大小和形状）
    （速度快-左上角右下角要历遍全图，中心点和宽高互不影响，不会因为一个座标点错而影响全局）

    4. 整个图以每个锚点为中心去画框（三种框），得到全图框后去和标定框算IOU，留下最好的框。
    5. 自己做训练的时候，要知道有多少个输出和框。比如我们有4个类，三种框，那么就是（5+4）*3


1. 网络结构：
    1.1 backbone：darknet53用于特征提取（卷积，下采样和残差块就不一一细说了）
    1.2 侦测网络用于检测
=====================================================
数据集准备：
    1. labelme或者labelimg等软件标注图片，得到xml，jason，txt-yolo格式都行。
    2. 重点是确保转成yolo的读取格式。图片，类别，中心点，框。

    3. pytorch准备数据集：
        __getitem__(pictures):这其实就是个for循环历遍每一张图。
            // N种不同大小的feature_size，比如论文中是13-26-52三种大小框。历遍三种维度。
            for feature_size in N:

                // 一张图上可能有多个框，历遍一张图上额的每个框。（这个可和上面的互换顺序都行）  
                for box in boxes:
                    原图上的c_x, c_y, w, h进行缩放等操作固定好尺寸。
                    计算中心点偏差和index，因为得到缩放后的两组参数后，在不同feature_size上可以去计算整数部分=index，小数部分就是中心点偏移量_x,_y
                    (上面就是计算出来了哪个cell/grid来负责检测目标。也得到了中心点偏移量)

                    // 一个feature_size map中每个grid/cell里面有多个或一个先验框，这个取决于你的类别数量（先验框聚类得到）
                    上面知道了是哪个cell来负责检测，每个grid里面有N个anchor box，下面就是要的到具体哪个anchor来负责检测（哪个先验框）
                    for pre_box in all_pre:
                        每个图片尺寸不一样，先固定到一个 M*M 固定尺寸；同时同比例缩放w，h，cx，cy
                        算cx，cy在每个 feature_size 上的位置。可以得到 整数具体index和中心点偏移量x_,y_
                        算 w和h  相对于  先验框w‘和h’的偏移量 p_w = w/w'; p_h = h/h'.
                        计算w，h，cx，cy偏差。以一个矩阵形式返回, 
      					(重点：搞清数据集的获取过程，组成，具体在matrix中组成形式)

        一个实际的dataset:[图片数量， 4， fs, fs, preBox, 5+NumClass]
        比如 》一张图4个数据为 "三个fs + 1个picData": 
            picData就是一个（3， 416， 416）矩阵。
            多个fs组成一个labels字典，比如13*13的检测框：
                labels[13]: [13, 13, 3, 8], [y_index, x_index, i, ??]；
                此处的i是，一个feature_size, 比如13*13，这个维度里面有三个先验框（3个class）
                ？？存的就是数据 [iox, c_x, c_y, log(p_w), log(p_h), one_hot(NumClass)],one_hot(NumClass)是个1*N的一维矩阵。

                1个fs会计算一次中心点偏移量，N个先验框(1个fs中)会有N个IOU，p_w, p_h.
                可以想象：13*13个像素格子，每个格子中有N条数据，一条是一个5+NumClass的一维数组。当然其他也一样，只不过其他都为0.

                一个13*13*N*8的全零初始化矩阵，只有1*1*N*8条是有数据的，其他都为0. 13*13个grid，每个grid里面都有N个anchor box(pre_box)
                对于训练图片中的ground truth，若其中心点落在某个cell内，那么该cell内的3个anchor box负责预测它，具体是哪个anchor box预测它，
                需要在训练中确定，即由那个与ground truth的IOU最大的anchor box预测它，而剩余的2个anchor box不与该ground truth匹配。
                YOLOv3需要假定每个cell至多含有一个grounth truth，而在实际上基本不会出现多于1个的情况。
                与ground truth匹配的anchor box计算坐标误差、置信度误差（此时target为1）以及分类误差，而其它的anchor box只计算置信度误差（此时target为0）。


    4. 
