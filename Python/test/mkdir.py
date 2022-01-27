import os
import sys

names = ["test_data_frontiermask", "test_data_frontiermask_freemask"]

for item in names:
    os.mkdir("/home/agent/test_dataset/input_8_44pth/"+item)
    for i in range(131, 141):
       os.mkdir("/home/agent/test_dataset/input_8_44pth/"+item+"/"+str(i))
       open("/home/agent/test_dataset/input_8_44pth/"+item+"/"+str(i)+"/map.txt", "w")
       open("/home/agent/test_dataset/input_8_44pth/"+item+"/"+str(i)+"/path.txt", "w")