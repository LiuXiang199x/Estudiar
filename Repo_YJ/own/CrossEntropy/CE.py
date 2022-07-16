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
    soft_input = torch.nn.Softmax(dim=0)
    print("Original inputs\n", input)
    print("After softmax\n", soft_input(input))

    #对softmax结果取log
    print(torch.log(soft_input(input)))

    
logSoftmax_NLLLoss()