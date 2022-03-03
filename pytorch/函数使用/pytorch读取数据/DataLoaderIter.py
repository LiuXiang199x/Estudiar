torch.utils.data.dataloader.DataLoaderIter

上面提到, DataLoaderIter就是DataLoaderIter的一个框架, 用来传给DataLoaderIter 一堆参数, 
并把自己装进DataLoaderIter 里.

其实到这里就可以满足大多数训练的需求了, 比如

class CustomDataset(Dataset):
   # 自定义自己的dataset

dataset = CustomDataset()
dataloader = Dataloader(dataset, ...)

for data in dataloader:
   # training...
   
   
在for 循环里, 总共有三点操作:

1. 调用了dataloader 的__iter__() 方法, 产生了一个DataLoaderIter
2. 反复调用DataLoaderIter 的__next__()来得到batch, 具体操作就是, 多次调用dataset的__getitem__()方法 
(如果num_worker>0就多线程调用), 然后用collate_fn来把它们打包成batch. 
中间还会涉及到shuffle , 以及sample 的方法等, 这里就不多说了.
3. 当数据读完后, __next__()抛出一个StopIteration异常, for循环结束, dataloader 失效.