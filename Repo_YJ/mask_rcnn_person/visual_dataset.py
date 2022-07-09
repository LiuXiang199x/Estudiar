from PIL import Image
import os
 
root_img_dir = "/home/agent/桌面/data/PennFudanPed/PNGImages"
root_mask_dir = "/home/agent/桌面/data/PennFudanPed/PedMasks"
 
Image.open(os.path.join(root_img_dir, 'FudanPed00001.png'))
 
mask = Image.open(os.path.join(root_mask_dir, 'FudanPed00001_mask.png'))
 
mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 255, 0, # index 2 is yellow
    255, 153, 0, # index 3 is orange
])
 
print(mask)
mask.show()