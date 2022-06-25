# 训练模型
def train():

    train_set = MyDataset(args.root + '/train', train=True)  # 训练数据对象
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)  # 将数据按batch size封装成Tensor
    # 选择使用的设备
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # 我是后来用cuda训练但是后来一直说内存不够，一气之下就将device直接改成cpu来训练了
    # 如果你没有我出现的内存不够的问题，还是直接用上上行的代码吧
    print(device)
    # 选择模型
    model = ResNet34()
    model.to(device)
    # 训练模式
    model.train()

    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 参数优化方法为Adam
    # 由命令行参数决定是否从之前的checkpoint开始训练
    if args.restore:
        checkpoint = torch.load(args.log_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
    else:
        loss = 0.0
        epoch = 0

    while epoch < args.epoch:
        running_loss = 0.0

        for i, data in tqdm(enumerate(train_loader), total=len(train_set)):  
            # tqdm是可以将for循环的过程用进度条表示出来
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            # 这里取出的数据就是 __getitem__() 返回的数据
            inputs, labels = data[0].to(device), data[1].to(device)  # 放入数据和标签

            # 根据pytorch中backward（）函数的计算，当网络参量进行反馈时，梯度是累积计算而不是被替换
            # 处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，因此需要对每个batch调用一遍zero_grad（）将参数梯度置0.
            optimizer.zero_grad()  # 将模型参数梯度初始化为0
            outs = model(inputs)  # 前向传播计算预测值
            loss = criterion(outs, labels)  # 计算当前损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新所有参数（用Adam）

            running_loss += loss.item()  # 累加每个batch里每个样本的loss

            if i % 200 == 199:  # 每训练200个样本就打印epoch，loss，并且保存checkpoint
                print('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, i + 1, running_loss / 200))  # 打印该loss平均值
                running_loss = 0.0
                # 保存 checkpoint
                print('Save checkpoint...')
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss},
                           args.log_path)

        epoch += 1

    print('Finish training')
    if i % 200 == 199:  # 每训练200个样本就打印epoch，loss，并且保存checkpoint
                    print('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, i + 1, running_loss / 200))  # 打印该loss平均值
                    running_loss = 0.0
                    # 保存 checkpoint
                    print('Save checkpoint...')
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss},
                            args.log_path)


# 验证模型，计算预测精度
def validation():

    # 测试数据集导入
    val_data = MyDataset(args.root + '/train', train=False)  # 验证集数据
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # 获取模型
    model = ResNet34()
    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 数据切换到测试模式

    with torch.no_grad():  # 停止gradient计算，从而节省了GPU算力和显存
        confusion_matrix = meter.ConfusionMeter(2)
        for ii, data in enumerate(val_dataloader):
            input, label = data  # 放入数据和标签
            score = model(input)
            _, pred = torch.max(score.data, 1)
        confusion_matrix.add(score.data, label.data)
        cm_value = confusion_matrix.value()  # 混淆矩阵
        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())  # 验证后模型准确率
    print('利用验证集得到模型准确率为: %.2f%%' % accuracy)

