import os

path = 'map_'	#设置创建后文件夹存放的位置
txt = "points_map"
for i in range(1, 76):	#这里创建10个文件夹
	isExists = os.path.exists(path+str(i))
	if not isExists:    #判断如果文件不存在,则创建
		os.makedirs(path+str(i))
		wenben = open(path+str(i) +"/"+txt+str(i)+".txt", "w") 
		print("%s 目录创建成功"%i)
	else:
        	print("%s 目录已经存在"%i)	
        	continue			

