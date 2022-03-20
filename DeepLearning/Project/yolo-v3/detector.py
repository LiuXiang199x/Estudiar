import os

import torch
from torch import nn
from yolo_v3_net import Yolo_V3_Net
from PIL import Image,ImageDraw
from utils import *
from dataset import *
from config import *

class_num={
    0:'person',
    1:'horse',
    2:'bicycle'
}
device = torch.device('cpu')
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        self.weights = 'params/net.pt'
        self.net = Yolo_V3_Net().to(device)
        if os.path.exists(self.weights):
            self.net.load_state_dict(torch.load(self.weights))
            print('加载权重成功')
        self.net.eval()

    def forward(self,input,thresh,anchors,case):
        output_13,output_26,output_52=self.net(input)
        index_13,bias_13=self.get_index_and_bias(output_13,thresh)
        boxes_13=self.get_true_position(index_13,bias_13,32,anchors[13],case)

        index_26, bias_26 = self.get_index_and_bias(output_26, thresh)
        boxes_26 = self.get_true_position(index_26, bias_26, 16, anchors[26],case)

        index_52, bias_52 = self.get_index_and_bias(output_52, thresh)
        boxes_52 = self.get_true_position(index_52, bias_52, 8, anchors[52],case)

        return torch.cat([boxes_13,boxes_26,boxes_52],dim=0)

    def get_index_and_bias(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)  # N,H,W,3,8

        mask = output[..., 0] > thresh  # N,H,W,3

        index = mask.nonzero()
        bias = output[mask]

        return index, bias

    def get_true_position(self,index,bias,t,anchors,case):
        anchors=torch.Tensor(anchors)

        a=index[:,3]

        cy=(index[:,1].float()+bias[:,2].float())*t/case
        cx=(index[:,2].float()+bias[:,1].float())*t/case

        w=anchors[a,0]*torch.exp(bias[:,3])/case
        h=anchors[a,1]*torch.exp(bias[:,4])/case

        p=bias[:,0]
        cls_p=bias[:,5:]
        cls_index=torch.argmax(cls_p,dim=1)

        return torch.stack([torch.sigmoid(p),cx,cy,w,h,cls_index],dim=1)


if __name__ == '__main__':
    detector=Detector()
    img=Image.open('images/1.jpg')
    _img=make_416_image('images/1.jpg')
    temp=max(_img.size)
    case=416/temp
    _img=_img.resize((416,416))
    _img=tf(_img).to(device)
    _img=torch.unsqueeze(_img,dim=0)
    results=detector(_img,0.000001,antors,case)
    draw=ImageDraw.Draw(img)

    for rst in results:
        x1,y1,x2,y2=rst[1]-0.5*rst[3],rst[2]-0.5*rst[4],rst[1]+0.5*rst[3],rst[2]+0.5*rst[4]
        print(x1,y1,x2,y2)
        print('class',class_num[int(rst[5])])
        draw.text((x1,y1),str(class_num[int(rst[5].item())])+str(rst[0].item())[:4])
        draw.rectangle((x1,y1,x2,y2),outline='red',width=1)
    img.show()