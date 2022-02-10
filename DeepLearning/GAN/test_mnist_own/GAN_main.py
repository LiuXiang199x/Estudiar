#coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  save_image
import os
#创建文件夹
if not os.path.exists('./img'):
    os.mkdir('./img')
def to_img(x):
    out=0.5*(x+1)
    out=out.clamp(0,1)#Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out=out.view(-1,1,28,28)#view()函数作用是将一个多行的Tensor,拼接成一行
    return out
batch_size=128
num_epoch=100
z_dimension=100
# 图像预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1,), (0.5,))
    ])
#mnist dataset mnist数据集下载
mnist=datasets.MNIST(
    root='./data/',train=True,transform=img_transform,download=True
)
#data loader 数据载入
dataloader=torch.utils.data.DataLoader(
    dataset=mnist,batch_size=batch_size,shuffle=True
)

#创建对象
D=discriminator()
G=generator()
if torch.cuda.is_available():
    D=D.cuda()
    G=G.cuda()
#########判别器训练train#####################
#分为两部分：1、真的图像判别为真；2、假的图像判别为假
#此过程中，生成器参数不断更新
#首先需要定义loss的度量方式  （二分类的交叉熵）
#其次定义 优化函数,优化函数的学习率为0.0003
criterion = nn.BCELoss() #是单目标二分类交叉熵函数
d_optimizer=torch.optim.Adam(D.parameters(),lr=0.0003)
g_optimizer=torch.optim.Adam(G.parameters(),lr=0.0003)
###########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch): #进行多个epoch的训练
    for i,(img, _) in enumerate(dataloader):
        num_img=img.size(0)
        # view()函数作用是将一个多行的Tensor,拼接成一行
        # 第一个参数是要拼接的tensor,第二个参数是-1
        # =============================训练判别器==================
        img = img.view(num_img, -1)  # 将图片展开为28*28=784
        real_img = Variable(img).cuda()  # 将tensor变成Variable放入计算图中
        real_label = Variable(torch.ones(num_img)).cuda()  # 定义真实的图片label为1
        fake_label = Variable(torch.zeros(num_img)).cuda()  # 定义假的图片的label为0
        # 计算真实图片的损失
        real_out = D(real_img)  # 将真实图片放入判别器中
        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
        real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好
        # 计算假的图片的损失
        z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 随机生成一些噪声
        fake_img = G(z)  # 随机噪声放入生成网络中，生成一张假的图片
        fake_out = D(fake_img)  # 判别器判断假的图片
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake #损失包括判真损失和判假损失
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数
        # ==================训练生成器============================
        ################################生成网络的训练###############################
        # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
        # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
        # 反向传播更新的参数是生成网络里面的参数，
        # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
        # 这样就达到了对抗的目的
        # 计算假的图片的损失
        z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 得到随机噪声
        fake_img = G(z) #随机噪声输入到生成器中，得到一副假的图片
        output = D(fake_img)  # 经过判别器得到的结果
        g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss
        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
        #打印中间的损失
        if (i+1)%100==0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                epoch,num_epoch,d_loss.data.item(),g_loss.data.item(),
                real_scores.data.mean(),fake_scores.data.mean()  #打印的是真实图片的损失均值
            ))
        if epoch==0:
            real_images=to_img(real_img.cpu().data)
            save_image(real_images, './img/real_images.png')
        fake_images = to_img(real_img.cpu().data)
        save_image(fake_images, './img/fake_images-{}.png'.format(epoch+1))
#保存模型
torch.save(G.state_dict(),'./generator.pth')
torch.save(D.state_dict(),'./discriminator.pth')
