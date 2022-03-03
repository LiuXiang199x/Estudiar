import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from resnet import resnet18



content = torch.load("/home/agent/Estudiar/DeepLearning/模型解析/models/resnet18+fc2/resnet18_latest_combined_v9.pth.tar")
content2 = torch.load("/home/agent/Estudiar/DeepLearning/模型解析/models/DRL/ckpt.999.pth")
# print(type(content))  # dict
# print(type(content2)) # dict

# print(len(content)) # 6
# print(len(content2)) # 2

# print(content.keys())   # dict_keys(['epoch', 'arch', 'model_state_dict', 'obj_state_dict', 'classifier_state_dict', 'best_prec1'])
# print(content2.keys()) # dict_keys(['global_state_dict', 'extra_state'])

# print(content["obj_state_dict"].keys())  # odict_keys(['module.fc.weight', 'module.fc.bias'])
# print(content["model_state_dict"].keys())
# odict_keys(['module.conv1.weight', 'module.bn1.weight', 'module.bn1.bias', 'module.bn1.running_mean', 'module.bn1.running_var', 'module.bn1.num_batches_tracked', 'module.layer1.0.conv1.weight', 'module.layer1.0.bn1.weight', 'module.layer1.0.bn1.bias', 'module.layer1.0.bn1.running_mean', 'module.layer1.0.bn1.running_var', 'module.layer1.0.bn1.num_batches_tracked', 'module.layer1.0.conv2.weight', 'module.layer1.0.bn2.weight', 'module.layer1.0.bn2.bias', 'module.layer1.0.bn2.running_mean', 'module.layer1.0.bn2.running_var', 'module.layer1.0.bn2.num_batches_tracked', 'module.layer1.1.conv1.weight', 'module.layer1.1.bn1.weight', 'module.layer1.1.bn1.bias', 'module.layer1.1.bn1.running_mean', 'module.layer1.1.bn1.running_var', 'module.layer1.1.bn1.num_batches_tracked', 'module.layer1.1.conv2.weight', 'module.layer1.1.bn2.weight', 'module.layer1.1.bn2.bias', 'module.layer1.1.bn2.running_mean', 'module.layer1.1.bn2.running_var', 'module.layer1.1.bn2.num_batches_tracked', 'module.layer2.0.

# print(content2["global_state_dict"].keys())
# # odict_keys(['actor_critic.actor.0.weight', 'actor_critic.actor.0.bias', 'actor_critic.actor.1.weight', 'actor_critic.actor.1.bias',.....])

# print(type(content["obj_state_dict"]))    # <class 'collections.OrderedDict'>
# print(type(content2["global_state_dict"]))  # <class 'collections.OrderedDict'>

# print(type(content["model_state_dict"]['module.conv1.weight']))  # <class 'torch.Tensor'>   # 64
# print(content["model_state_dict"]['module.conv1.weight'].shape)  # <class 'torch.Tensor'>   # torch.Size([64, 3, 7, 7])
# print(type(content2["global_state_dict"]["actor_critic.actor.0.weight"]))   # <class 'torch.Tensor'> # 8
# print(content2["global_state_dict"]["actor_critic.actor.0.weight"].shape)  # torch.Size([8, 4, 5, 5])

# .pth / .pth.tar ===> {"key":value} ===> value=<class 'collections.OrderedDict'>看源码，该class继承的dict，所以还是一个dict
# {"key":class:dict("keys":tensor)}   # 三层字典


# 只保存模型用于以后的推断的话使用.pth或.pt，这样可以直接加载模型
# A common PyTorch convention is to save models using either a .pt or .pth file extension.

# torch.save(model, "model.pth") # or .pt
# model = torch.load("model.pth")

# 断点保存的话则使用.tar，加载的时候模型需要使用load_state_dict()方法
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             ...
#             }, "checkpoint.tar")

# ...

# checkpoint = torch.load("checkpoint.tar")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']