from torch.utils.data import Dataset
# torch.utils.data.Dataset
# 是一个抽象类, 自定义的Dataset需要继承它并且实现两个成员方法:
    # __getitem__()
    # __len__()
# 第一个最为重要, 即每次怎么读数据. 以图片为例:


第一个最为重要, 即每次怎么读数据. 以图片为例:

    def __getitem__(self, index):
        img_path, label = self.data[index].img_path, self.data[index].label
        img = Image.open(img_path)

        return img, label
# 这里有一个__getitem__函数，__getitem__函数接收一个index，
# 然后返回图片数据和标签，这个index通常是指一个list的index，
# 这个list的每个元素就包含了图片数据的路径和标签信息。
    
    
值得一提的是, pytorch还提供了很多常用的transform, 在torchvision.transforms 里面常用的有Resize , RandomCrop , 
Normalize , ToTensor (这个极为重要, 可以把一个PIL或numpy图片转为torch.Tensor, 
但是好像对numpy数组的转换比较受限, 所以这里建议在__getitem__()里面用PIL来读图片, 而不是用skimage.io).

第二个比较简单, 就是返回整个数据集的长度:

    def __len__(self):
        return len(self.data)