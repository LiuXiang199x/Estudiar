import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


# https://www.cnblogs.com/monologuesmw/p/13686442.html

image = imageio.imread("/home/agent/Estudiar/DeepLearning/ImgAugmenters/bedroom.jpg")
crop = iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge")
img_aug = crop(image=image)
print("Original")
ia.imshow(np.hstack([image, img_aug]))
#ia.imshow(image)
#ia.imshow(img_aug)