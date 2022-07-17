from cv2 import log
import torchvision
import torch
import torch.nn as nn

def testSoftmaxSigmoid():
    activation = nn.Softmax(dim=0)
    activation1 = nn.LogSoftmax(dim=0)

    activation2 = nn.Sigmoid()
    activation3 = nn.LogSigmoid()


    inputs = torch.tensor([1., 2.])
    print("===========> test for sigmoid")
    print(activation2(inputs))
    print(1/(1+torch.exp(torch.tensor(-1.))))
    print(1/(1+torch.exp(torch.tensor(-2.))))

    print("===========> test for LogSigmoid: LogSigmoid = log(Sigmoid)")
    print(activation3(inputs))
    print(torch.log(1/(1+torch.exp(torch.tensor(-1.)))))
    print(torch.log(1/(1+torch.exp(torch.tensor(-2.)))))

    print("===========> test for softmax")
    print(activation(inputs))
    print(torch.exp(torch.tensor(1.))/(torch.exp(torch.tensor(1.))+torch.exp(torch.tensor(2.))))
    print(torch.exp(torch.tensor(2.))/(torch.exp(torch.tensor(1.))+torch.exp(torch.tensor(2.))))

    print("===========> test for logSoftmax: LogSoftmax = log(softmax)")
    print(activation1(inputs))
    print(torch.log(torch.exp(torch.tensor(1.))/(torch.exp(torch.tensor(1.))+torch.exp(torch.tensor(2.)))))
    print(torch.log(torch.exp(torch.tensor(2.))/(torch.exp(torch.tensor(1.))+torch.exp(torch.tensor(2.)))))


def logSoftmax_NLLLoss():
    print("=========> NLLLoss: Negative log likelihood Loss")
    print("=========> softmax(x)+log(x)+NLLLoss = CrossEntropy \
        logSoftmax+NLLLoss = CrossEntropy")
    print("NLLLoss是把logSigmoid/logSoftmax的输出与label对应的值拿出来，去掉负号再求均值。")

    input=torch.randn(3,3)
    soft_input = torch.nn.Softmax(dim=1)
    print("Original inputs\n", input)
    print("After softmax\n", soft_input(input))

    #对softmax结果取log
    print("torch.log(soft_input(input)): \n", torch.log(soft_input(input)))
    print("假设labels为：[0, 1, 2]，第一行取第0，第二行取出1...")
    labels = torch.tensor([0, 1, 2])
    Loss1 = torch.abs(torch.log(soft_input(input))[0][0]) + \
            torch.abs(torch.log(soft_input(input))[1][1]) + \
            torch.abs(torch.log(soft_input(input))[2][2])
    print("计算出logsofmax后，根据labels[0,1,2]找出没一行对应的值相加后求平均，Loss = ", Loss1/len(labels))

    activation2 = nn.NLLLoss()
    print("log_softmax + NLLLoss: ", activation2(torch.log(soft_input(input)), labels))
    
    activation3 = nn.CrossEntropyLoss()
    print("CrossEntropy loss: ", activation3(input, labels))

logSoftmax_NLLLoss()
# 总结：crossEntropy就是 CE = -Y(gt)*logY(p); 
    # 所以在拆解计算的时候，inputs直接进行softmax按行1/列0得到概率值
    # 然后对softmax的值都进行log ===> 相当于完成了 logY(p)
    # 再根据GT中的labels比如这里是[0, 1, 2], 选出对应的相加求平均就行
    #  tensor([[-1.2030, -0.4973, -2.3911],
    #     [-0.7096, -1.4014, -1.3398],
    #     [-1.7054, -0.7318, -1.0868]])
        # 这一步详细说明，比如0，代表第一行的GT是第一个元素，所以Y(gt)=1,但是在求出来的里面是 -1.2(比2.39小,但是没办法,说明预测错了)
        # 但是还是继续按照公式来算, 0代表第0位为GT概率是1,所以是 -1*(-1.0);1代表第二行GT是第二位概率为1,-1*(-1.4),其实按照0,1,2都预测错误了,因为softmax后可以看到概率最大的是1,0,1
        # 宗上就是 (-1)*(-1.2) + (-1)*(-1.4) + (-1)*(-1.08) 然后求平均就得到最终Loss了; softmax概率越大,越接近1,那么log后越接近0!!!!,才对的越多loss越小.
    # 转log好处： softmax是 0-1，GT为1，概率越大越接近1，相乘后猜的越对越接近1，loss就越来越大了。log在0-1上，自变量越接近1，log后越接近0，所以做一个log转换。log(softmax)
#