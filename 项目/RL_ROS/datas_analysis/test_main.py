from FBE_datasets import Datasets

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


class Visualization_data():
    def __init__(self):
        super().__init__
        self.datasets = Datasets()


    def histogram(self):

        label_list = []

        for i in range(10):
            label_list.append("map_"+str(i))

        tmp_area, tmp_path = self.datasets.get_sum()
        data_list = []
        print("tmp_area:", tmp_area)
        print("tmp_path:", tmp_path)
        for i in range(10):
            data_list.append((tmp_path[i]/tmp_area[i])/0.05)

        print("data_list:", data_list)

        x = label_list
        plt.bar(x, height=data_list, width=0.6, alpha=0.8, color='blue')

        plt.ylim(0, max(data_list))     # y轴取值范围
        plt.ylabel("TotalPath/TotalArea")
 
        # plt.xticks([index + 0.2 for index in x], label_list)
        plt.xlabel("TestMaps")
        plt.title("Cost of length path(m) per unit area(m^2)")
        plt.legend()  
        
     
     
        plt.show()

Visualization_data().histogram()