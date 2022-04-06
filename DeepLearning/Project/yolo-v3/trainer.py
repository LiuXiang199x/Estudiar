import os

from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from dataset import *
from yolo_v3_net import *
from torch.utils.tensorboard import SummaryWriter

def loss_fun(output, target, c):
    # permute的中文意思是，“排列组合”。permute()可以对某个张量的任意维度进行调换。
    # # torch.Size([2, 45, 13, 13]),  torch.Size([2, 13, 13, 3, 8])
    # output.permute: ([2, 13, 13, 45])
    output = output.permute(0, 2, 3, 1)
    
    # ([2, 13, 13, 45]) --> ([2, 13, 13, 3, 15]) --> 15包括三个anchor box
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    # iou_scores：真实值与最匹配的anchor的IOU得分值 class_mask：分类正确的索引  
    # obj_mask：目标框所在位置的最好anchor置为1 noobj_mask obj_mask那里置0，
    # 还有计算的iou大于阈值的也置0，其他都为1 tx, ty, tw, th, 
    # 对应的对于该大小的特征图的xywh目标值也就是我们需要拟合的值 tconf 目标置信度
    # loss可分为loss_obj, loss_noobj, loss_cls, loss_coor4个部分边界框坐标损失、置信度损失和分类损失
    mask_obj = target[..., 0] > 0
    mask_no_obj = target[..., 0] == 0  # 判断当前网格中是否检测到物体，有为True
    # print("mask_obj.shape", mask_obj.shape)  # (1, 13, 13 ,3)
    # print("mask_np_obj.shape", mask_no_obj.shape)  # (1, 13, 13, 3)
    
    # 用的时候需要在该层前面加上Sigmoid函数，因为只有正例和反例，且两者的概率和为 1，那么只需要预测一个概率就好了
    loss_p_fun = nn.BCELoss()

    # 这个公式tx,ty为何要sigmoid一下啊？前面讲到了在yolov3中没有让Gx - Cx后除以Pw得到tx，而是直接Gx - Cx得到tx，
    # 这样会有问题是导致tx比较大且很可能>1.(因为没有除以Pw归一化尺度)。用sigmoid将tx,ty压缩到[0,1]区间內，
    # 可以有效的确保目标中心处于执行预测的网格单元中，防止偏移过多。
    loss_p = loss_p_fun(torch.sigmoid(output[..., 0]), target[..., 0])


    # 坐标的损失。有目标的像素格子才是我们要考虑的
    loss_box_fun = nn.MSELoss()
    loss_box = loss_box_fun(output[mask_obj][..., 1:5], target[mask_obj][..., 1:5])

    # 分类损失：
    loss_segment_fun = nn.CrossEntropyLoss()
    loss_segment = loss_segment_fun(output[mask_obj][..., 5:],
                                    torch.argmax(target[mask_obj][..., 5:], dim=1, keepdim=True).squeeze(dim=1))

    loss = c * loss_p + (1 - c) * 0.5 * loss_box + (1 - c) * 0.5 * loss_segment
    return loss

if __name__ == '__main__':
    summary_writer=SummaryWriter('logs')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = YoloDataSet()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # print(data_loader)
    # print(type(data_loader))  # <class 'torch.utils.data.dataloader.DataLoader'>

    # weight_path = 'params/net.pt'
    net = Yolo_V3_Net().to(device)
    # if os.path.exists(weight_path):
    #    net.load_state_dict(torch.load(weight_path))

    opt = optim.Adam(net.parameters())

    #
    epoch=0
    index = 0
    while True:
        for target_13, target_26, target_52, img_data in data_loader:
            target_13, target_26, target_52, img_data = target_13.to(device), target_26.to(device), target_52.to(
                device), img_data.to(device)

            # 一个target = 标注图片与N个先验框的差
            print("target_13: ", target_13.size())  # torch.Size([1, 13, 13, 3, 8])
            print("img_data: ", img_data.size())   # torch.Size([1, 3, 416, 416])
            output_13, output_26, output_52 = net(img_data)
            
            # print(output_13.size())  # torch.Size([1, 45, 13, 13])
            # print(output_26.size())  # torch.Size([1, 45, 13, 13])
            # print(output_52.size())  # torch.Size([1, 45, 13, 13])
            # print(target_13.size())  # torch.Size([1, 13, 13, 3, 8])
            
            # 0.7 作为NMS阈值
            loss_13 = loss_fun(output_13.float(), target_13.float(), 0.7)
            loss_26 = loss_fun(output_26.float(), target_26.float(), 0.7)
            loss_52 = loss_fun(output_52.float(), target_52.float(), 0.7)

            loss = loss_13 + loss_26 + loss_52
            print("loss_13: ", loss_13)
            print("loss_26: ", loss_26)
            print("loss_52: ", loss_52)
            print("loss: ", loss)
            break
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'loss{epoch}=={index}',loss.item())
            summary_writer.add_scalar('train_loss',loss,index)
            index+=1

        # torch.save(net.state_dict(),'/home/agent/Estudiar/DeepLearning/Project/yolo-v3/params/net.pt')
        # print('模型保存成功')
        # epoch+=1
        break