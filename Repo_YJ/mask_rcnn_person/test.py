from calendar import month_abbr
from statistics import mode
from coco_eval import *
from coco_utils import *
from utils import *
from load_dataset import *
from maskrcnn_api import *
import transforms as T
from engine import train_one_epoch, evaluate

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
 
    return T.Compose(transforms)

# root_dir = "/home/agent/桌面/data/PennFudanPed"
root_dir = "/home/agent/桌面/data/slam_maps/room_maps_train/fudan_format"

a = PennFudanDataset(root_dir)
dataset = PennFudanDataset(root_dir, get_transform(train=True))
dataset_test = PennFudanDataset(root_dir, get_transform(train=False))
 
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
 
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
 
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

print(data_loader)