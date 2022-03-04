import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from resnet import resnet18


content = torch.load("/home/agent/Estudiar/DeepLearning/模型解析/models/resnet18+fc2/resnet18_latest_combined_v9.pth.tar")
content2 = torch.load("/home/agent/Estudiar/DeepLearning/模型解析/models/DRL/ckpt.999.pth")

f_cobj_weight = open("/home/agent/Estudiar/DeepLearning/模型解析/models/pramas/obj_fc_weight.txt", "w")
f_cobj_bias = open("/home/agent/Estudiar/DeepLearning/模型解析/models/pramas/obj_fc_bias.txt", "w")
f_cclass_weight = open("/home/agent/Estudiar/DeepLearning/模型解析/models/pramas/class_fc_weight.txt", "w")
f_cclass_bias = open("/home/agent/Estudiar/DeepLearning/模型解析/models/pramas/class_fc_bias.txt", "w")
f_test = open("/home/agent/Estudiar/DeepLearning/模型解析/models/pramas/test.txt", "w")

def save_params():
    c = content
    print("c.keys():", c.keys())
    c_obj = c['obj_state_dict']
    c_class = c['classifier_state_dict']
    print(c_obj.keys())
    print(c_class.keys())
    print(c_class['module.fc.bias'])   
    print(c_obj['module.fc.bias'])
    print("c_obj['module.fc.weight'].shape:", c_obj['module.fc.weight'].shape)  # c_obj['module.fc.weight'].shape: torch.Size([512, 13])
    print("c_class['module.fc.weight'].shape:", c_class['module.fc.weight'].shape)  # c_class['module.fc.weight'].shape: torch.Size([5, 1024])
    print("c_obj['module.fc.bias'].shape:", c_obj['module.fc.bias'].shape) # c_obj['module.fc.bias'].shape: torch.Size([512])
    print("c_class['module.fc.bias'].shape:", c_class['module.fc.bias'].shape) # c_class['module.fc.bias'].shape: torch.Size([5])
    
    for i in c_class['module.fc.bias']:
        print(i.item())
        f_cclass_bias.writelines(str(i.item())+"\n")

    for i in c_obj['module.fc.bias']:
        f_cobj_bias.writelines(str(i.item())+"\n")
        
    for i in c_class['module.fc.weight']:\
        # f_cclass_weight.writelines(str(i.cpu().numpy())+"\n")
        counter_class = 0
        for item in i.cpu().numpy():
            f_cclass_weight.write(str(item))
            counter_class += 1
            if counter_class<1024:
                f_cclass_weight.write(",") 
            else:
                f_cclass_weight.write("\n")
                counter_class = 0
            
    for i in c_obj['module.fc.weight']:
        # f_cobj_weight.writelines(str(i.cpu().numpy())+"\n")
        counter_obj = 0
        for item in i.cpu().numpy():
            f_cobj_weight.write(str(item))
            counter_obj += 1
            if counter_obj<13:
                f_cobj_weight.write(",") 
            else:
                f_cobj_weight.write("\n")
                counter_obj = 0
             
save_params()

###### torch.load 把网络中的{{{}}}数据读出来 ######### 然后 model.load_state_dict() 把torch.load读出来的东西加载到网络上去
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
