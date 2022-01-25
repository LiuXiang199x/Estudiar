from FBE_datasets import Datasets_FBE
from Input8_F import Datasets_8F
from Input8_FF import Datasets_8FF
from Input4_F import Datasets_4F
from Input4_FF import Datasets_4FF

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


class Visualization_data():
    def __init__(self):
        super().__init__
        self.datasets_fbe = Datasets_FBE()
        self.datasets_8if = Datasets_8F()
        self.datasets_8iff = Datasets_8FF()
        self.datasets_4if = Datasets_4F()
        self.datasets_4iff = Datasets_4FF()

    def histogram(self):

        label_list = []

        for i in range(10):
            label_list.append("map_"+str(i))

        tmp_real = self.datasets_fbe.return_realarea()
        tmp_area_fbe, tmp_path_fbe = self.datasets_fbe.get_sum()
        tmp_area_fbe, tmp_path_fbe = self.datasets_fbe.get_sum()
        tmp_area_fbe, tmp_path_fbe = self.datasets_fbe.get_sum()
        tmp_area_fbe, tmp_path_fbe = self.datasets_fbe.get_sum()
        tmp_area_fbe, tmp_path_fbe = self.datasets_fbe.get_sum()



        data_list = []


        print("data_list:", data_list)

        x = label_list
        plt.bar(x, height=data_list, width=0.6, alpha=0.8, color='blue')

        plt.ylim(0, max(data_list))     # y轴取值范围
        # plt.ylabel("TotalPath/TotalArea")
 
        # plt.xticks([index + 0.2 for index in x], label_list)
        # plt.xlabel("TestMaps")
        # plt.title("Cost of length path(m) per unit area(m^2)")
        plt.legend()  
        
     
     
        plt.show()

Visualization_data().histogram()