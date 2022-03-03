import resnet
import torch
import torch.nn as nn

class Object_Linear(nn.Module):
    def __init__(self):
        super(Object_Linear, self).__init__()
        self.fc = nn.Linear(13, 512)

    def forward(self, x):
        out = self.fc(x)
        return out


class LinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LinClassifier, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, conv, idt):
        out = torch.cat((conv, idt), 1)
        out = self.fc(out)
        return out


class AllModel(nn.Module):
    def __init__(self, num_classes):
        super(AllModel, self).__init__()
        self.num_classes = num_classes
        
        self.classifier = LinClassifier(num_classes)
       
        self.object_idt = Object_Linear()
        model = resnet.resnet18(num_classes=num_classes) # to be commented
        self.model = model

    def forward(self, x_rgb, x_onehot):
        output_conv = self.model(x_rgb)  # 1*512
        output_idt1 = self.object_idt(x_onehot)  # 1*512
        output_idt2 = output_idt1.unsqueeze(0)
        output = self.classifier(output_conv, output_idt2)
        return output