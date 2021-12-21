import os

path = 'map_'
txt = "points_map"
for i in range(76, 127):	
	isExists = os.path.exists(path+str(i))
	if not isExists:    
		os.makedirs(path+str(i))
		wenben = open(path+str(i) +"/"+txt+str(i)+".txt", "w") 
		print("%s 目录创建成功"%i)
	else:
        	print("%s 目录已经存在"%i)	
        	continue			

