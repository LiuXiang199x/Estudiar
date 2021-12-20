#!/usr/bin/env python
#-*- coding:UTF-8 -*-
import os
import re
import shutil
        
path ='/home/xiang/Estudiar/dataset/maps/original_maps_800*800/png/' 
f=open("allpic_name.txt")

# count = 0
# for pic in os.listdir(path):
#	count += 1
#	f.write(os.path.join(path,pic)+'\n')    


filelist = os.listdir(path)
count = 1
for item in filelist:
    # print('item name is ',item)
    if item.endswith('.png'):
        name = item.split('.',1)[0]
        src = os.path.join(os.path.abspath(path), item)
        dst = os.path.join(os.path.abspath(path), "map_" + str(count) + '.png')
    try:
        os.rename(src, dst)
        print('rename from %s to %s' %(src, dst))
    except:
        continue
    count += 1

lenl = len(filelist)
for i in range(lenl):
    #if i.find('testnan')!=-1:
    i += 1
    shutil.move(path + "map_" + str(i) + ".png","/home/xiang/Estudiar/dataset/maps/file_est/map_"+ str(i) +"/", "map_"+str(i) +".png")
    #print(i,name[i])
