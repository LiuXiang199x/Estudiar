import torchvision
import torch
import torch.nn as nn

def NLLLoss():
    print("=========> softmax(x)+log(x)+NLLLoss = CrossEntropy")

    activation = nn.NLLLoss()

    inputs = torch.tensor([1., 2.])
    print("===========> test for sigmoid")
    print(activation2(inputs))
    print(1/(1+torch.exp(torch.tensor(-1.))))
    print(1/(1+torch.exp(torch.tensor(-2.))))

    print("===========> test for softmax")
    print(activation(inputs))
    print(torch.exp(torch.tensor(1.))/(torch.exp(torch.tensor(1.))+torch.exp(torch.tensor(2.))))
    print(torch.exp(torch.tensor(2.))/(torch.exp(torch.tensor(1.))+torch.exp(torch.tensor(2.))))

