import torch
from torchvision.transforms import transforms
from PIL import Image

# transforms.Compose()方法接收一个 transforms 方法的list为参数，
# 将这些操作组合到一起，返回一个新的tranforms。
# 通常用于包装一个完整的变换操作的pipeline

tf1=transforms.Compose([
    transforms.CenterCrop(400),
])

tf2=transforms.Compose([
    transforms.Resize(400),
])

tf3=transforms.Compose([
    transforms.Resize(400),
    transforms.ToTensor(),
])


def get_images(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        imgs_info = f.readlines()
        imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
    return imgs_info


datas = get_images("/home/agent/Estudiar/pytorch/函数使用/datas.txt")
print(datas)
for item in datas:
    img_path = item[0]
    label = item[1]
    print(img_path)
    print(label)
    imgg = Image.open(img_path)
    print(type(imgg))    
    print(imgg)
    # imgg.show()
    print(imgg.size)
    
    img2 = tf2(imgg)
    print(img2)
    img2.show()
    
    
    break