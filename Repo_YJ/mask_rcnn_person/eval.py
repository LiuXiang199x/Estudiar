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

root_dir = "/home/agent/桌面/data/PennFudanPed"

# use the PennFudan dataset and defined transformations
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
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 
# the dataset has two classes only - background and person
num_classes = 2
save_dic = "/home/agent/桌面/model_dicts.pth"

# get the model using the helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.load_state_dict(torch.load(save_dic))
model.to(device)
model.eval()

img,_ = dataset_test[2]

with torch.no_grad():
    prediction = model([(img.to(device))])

print(prediction[0].keys())
print(len(prediction[0]["masks"]))
Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).show()
# for i in range(len(prediction[0]["masks"])):
    
#     Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()).show()

