import os
import sys

names = ["frontier", "frontier+freemask"]

for item in names:
    os.mkdir("/home/agent/test_dataset/input_4/"+item)
    for i in range(131, 141):
       os.mkdir("/home/agent/test_dataset/input_4/"+item+"/"+str(i))
       open("/home/agent/test_dataset/input_4/"+item+"/"+str(i)+"/map.txt", "w")
       open("/home/agent/test_dataset/input_4/"+item+"/"+str(i)+"/path.txt", "w")