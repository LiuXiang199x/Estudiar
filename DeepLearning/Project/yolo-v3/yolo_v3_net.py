import torch
from torch import nn
from torch.nn import functional


# 卷积块
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.sub_module(x)


# 残差块
class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.sub_module(x) + x


# 卷积集合块
class ConvolutionalSetLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSetLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.sub_module(x)


# 下采样
class DownSamplingLayer(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(DownSamplingLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channel, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


# 上采样
class UpSamplingLayer(nn.Module):
    def __init__(self):
        super(UpSamplingLayer, self).__init__()

    def forward(self, x):
        return functional.interpolate(x, scale_factor=2, mode='nearest')


class Yolo_V3_Net(nn.Module):
    def __init__(self):
        super(Yolo_V3_Net, self).__init__()

        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSamplingLayer(32, 64),

            ResidualLayer(64, 32),

            DownSamplingLayer(64, 128),

            ResidualLayer(128, 64),
            ResidualLayer(128, 64),

            DownSamplingLayer(128, 256),

            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128)
        )

        self.trunk_26 = nn.Sequential(
            DownSamplingLayer(256, 512),

            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256)
        )

        self.trunk_13 = nn.Sequential(
            DownSamplingLayer(512, 1024),

            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512)
        )

        # 13*13 就是去检测大目标的，因为图像分成13*13那么每个块就会很大
        self.convset_13 = nn.Sequential(
            ConvolutionalSetLayer(1024, 512)
        )
        self.detetion_13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 45, 1, 1, 0)
        )

        self.up_13_to_26 = nn.Sequential(
            ConvolutionalLayer(512, 256, 3, 1, 1),
            UpSamplingLayer()
        )

        self.convset_26 = nn.Sequential(
            ConvolutionalSetLayer(768, 256)
        )
        self.detetion_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 45, 1, 1, 0)
        )
        self.up_26_to_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 3, 1, 1),
            UpSamplingLayer()
        )

        self.convset_52 = nn.Sequential(
            ConvolutionalSetLayer(384, 128)
        )
        self.detetion_52 = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 45, 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_13_out = self.convset_13(h_13)
        detetion_13_out = self.detetion_13(convset_13_out)
        up_13_to_26_out = self.up_13_to_26(convset_13_out)
        cat_13_to_26 = torch.cat((up_13_to_26_out, h_26), dim=1)

        convset_26_out = self.convset_26(cat_13_to_26)
        detetion_26_out = self.detetion_26(convset_26_out)
        up_26_to_52_out = self.up_26_to_52(convset_26_out)
        cat_26_to_52 = torch.cat((up_26_to_52_out, h_52), dim=1)

        convset_52_out = self.convset_52(cat_26_to_52)
        detetion_52_out = self.detetion_52(convset_52_out)

        return detetion_13_out, detetion_26_out, detetion_52_out


if __name__ == '__main__':
    net = Yolo_V3_Net()
    net.load_state_dict(torch.load("/home/agent/Estudiar/DeepLearning/Project/yolo-v3/params/net.pt"))
    x = torch.randn(1, 3, 416, 416)
    y = net(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    torch.save(net, "/home/agent/Estudiar/DeepLearning/Project/yolo-v3/params/net_all.pkl")
    torch.save(net.state_dict(), "/home/agent/Estudiar/DeepLearning/Project/yolo-v3/params/net_params.pth")