import numpy as np
import os

file_path = "/home/agent/test_dataset/input_8/test_data_frontiermask_freemask/"

class Datasets():
    def __init__(self):
        self.map_exp_0 = []
        self.map_path_0 =[]
        
        self.map_exp_1 = []
        self.map_path_1 =[]
        
        self.map_exp_2 = []
        self.map_path_2 =[]
        
        self.map_exp_3 = []
        self.map_path_3 =[]
        
        self.map_exp_4 = []
        self.map_path_4 =[]
        
        self.map_exp_5 = []
        self.map_path_5 =[]
        
        self.map_exp_6 = []
        self.map_path_6 =[]
        
        self.map_exp_7 = []
        self.map_path_7 =[]
        
        self.map_exp_8 = []
        self.map_path_8 =[]
        
        self.map_exp_9 = []
        self.map_path_9 =[]

        self.initialization()

    def initialization(self):
        for i in range(131, 141):
            file = open(file_path+str(i)+"/map.txt")
            file2 = open(file_path+str(i)+"/path.txt")
            if i==131:
                for item in file:
                    self.map_exp_0.append(int(item))
                for item2 in file2:
                    self.map_path_0.append(eval(item2))
                    
            if i==132:
                for item in file:
                    self.map_exp_1.append(int(item))
                for item2 in file2:
                    self.map_path_1.append(eval(item2))

            if i==133:
                for item in file:
                    self.map_exp_2.append(int(item))
                for item2 in file2:
                    self.map_path_2.append(eval(item2))

            if i==134:
                for item in file:
                    self.map_exp_3.append(int(item))
                for item2 in file2:
                    self.map_path_3.append(eval(item2))

            if i==135:
                for item in file:
                    self.map_exp_4.append(int(item))
                for item2 in file2:
                    self.map_path_4.append(eval(item2))

            if i==136:
                for item in file:
                    self.map_exp_5.append(int(item))
                for item2 in file2:
                    self.map_path_5.append(eval(item2))

            if i==137:
                for item in file:
                    self.map_exp_6.append(int(item))
                for item2 in file2:
                    self.map_path_6.append(eval(item2))

            if i==138:
                for item in file:
                    self.map_exp_7.append(int(item))
                for item2 in file2:
                    self.map_path_7.append(eval(item2))

            if i==139:
                for item in file:
                    self.map_exp_8.append(int(item))
                for item2 in file2:
                    self.map_path_8.append(eval(item2))

            if i==140:
                for item in file:
                    self.map_exp_9.append(int(item))
                for item2 in file2:
                    self.map_path_9.append(eval(item2))

a = Datasets()
print(a.map_exp_0)
print(a.map_path_0)
a= [1,2,3,4]
print(a/[2,2,2,2])