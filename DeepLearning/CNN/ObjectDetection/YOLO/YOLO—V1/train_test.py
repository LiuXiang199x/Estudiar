from modulefinder import IMPORT_NAME
import torch
import torch.nn as nn
from VGG import VGG
from YOLO_V1 import YOLOV1

def train():
    for epoch in range(epochs):
        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # 取图片
            inputs = input_process(batch)
            # 取标注
            labels = target_process(batch)
            
            # 获取得到输出
            outputs = yolov1_model(inputs)
            #import pdb
            #pdb.set_trace()
            #loss = criterion(outputs, labels)
            loss,lm,glm,clm = lossfunc_details(outputs,labels)
            loss.backward()
            optimizer.step()
            #print(torch.cat([outputs.detach().view(1,5),labels.view(1,5)],0).view(2,5))
            if iter % 10 == 0:
            #    print(torch.cat([outputs.detach().view(1,5),labels.view(1,5)],0).view(2,5))
                print("epoch{}, iter{}, loss: {}, lr: {}".format(epoch, iter, loss.data.item(),optimizer.state_dict()['param_groups'][0]['lr']))
        
        #print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        #print("*"*30)
        #val(epoch)
        scheduler.step()