from torch.utils.data import DataLoader
@torch.utils.data.DataLoader

class torch.utils.data.DataLoader(dataset, batch_size=1, 
                                  shuffle=False, sampler=None, batch_sampler=None, 
                                  num_workers=0, collate_fn=<function default_collate>, 
                                  pin_memory=False, drop_last=False)
可以看到, 主要参数有这么几个:

dataset : 即上面自定义的dataset.
collate_fn: 这个函数用来打包batch, 后面详细讲.
num_worker: 非常简单的多线程方法, 只要设置为>=1, 就可以多线程预读数据啦.
这个类其实就是下面将要讲的DataLoaderIter的一个框架, 一共干了两件事: 
    1.定义了一堆成员变量, 到时候赋给DataLoaderIter, 
    2.然后有一个__iter__() 函数, 把自己 "装进" DataLoaderIter 里面.

def __iter__(self):
        return DataLoaderIter(self)