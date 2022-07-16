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