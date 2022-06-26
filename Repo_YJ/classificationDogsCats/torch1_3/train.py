import torch
import torch.nn as nn
from datasets_laptop import loadDatasTrain
from Network import Tmodel


EPOCHES = 100000

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tmodel().to(device)
    datas = loadDatasTrain()
    
    criterion = nn.CrossEntropyLoss()
    optm = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(EPOCHES):
        print("=========> epoch/EPOCHS: {}/{}".format(epoch, EPOCHES))
        runningLoss = 0
        
        for iter_num, data in enumerate(datas, 0):
            data_, label_ = data
            data_ = data_.to(device)
            label_ = label_.to(device)

            optm.zero_grad()
            predicction = model(data_)
            loss_ = criterion(predicction, label_)
            loss_.backward()
            optm.step()

            runningLoss += loss_

            if epoch%5 == 0:
                print("running loss = ", runningLoss)

        if epoch % 1000 == 0:
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict()},
                "/home/marco/Estudiar/Repo_YJ/classificationDogsCats/torch1_3/model_{}.pth".format(epoch))

if __name__ == "__main__":
    train()