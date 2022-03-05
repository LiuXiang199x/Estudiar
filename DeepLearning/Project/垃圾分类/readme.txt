第一大步：
    训练开始前：图片处理，分组。
    训练开始中：图片打包。分别生成想要的train，test，val列表。比如train两个列表train_url放图片位置，train_label放标签位置。
                可以用transform来批量处理数据（旋转，剪裁等，用它来将img变成tensor）
                Dataset：将所有数据打包，进行transform等操作，预处理做完。全部打包成((tensor[], label), ....)形式。
                Dataloader：将以上数据进行分组，batchsize等操作，算力选择等，url和label全部都打包成tensor准备送入网络训练。

第二大步：
    网络准备，定义网络，初始化网络（要不要预训练，或者加载本地权重等）
    网络搭到gpu上面等操作

第三大步：
    定义各种参数和方程：
        损失函数，优化器，可视化log，损失系数。
        epoches，batch_size，num_classes等。

第四大步：
    开始训练：
        先step()清空梯度信息。step()和backward()是不一样的。
        opt.step()先更新网络参数，backward()更新梯度。
        一个epoch中train()一次和val()一次
            train():
                初始化所有值：loss，top1，2，3..， 任何你想定义的。
                一次train用一批数据：train_loader(比如我一次train用1000，一个batch_size=10)
                所以我一个epoch里面1000个数据，80个epoch的话，自动每个epoch 1000个数据（假设总共80000数据）
                一次epoch中1000个数据，已俄国batch_size=10，那么就是train100次

            val()：
                具体流程和train()是类似的。
        
            是否要保存一轮模型(涉及到和上一轮测试结果对比) ，哪些单独保存，哪几层单独保存{}{}{}都是以字典形式保存。
