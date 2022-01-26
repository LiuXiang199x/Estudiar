from turtle import width
from cv2 import mean

from matplotlib.colors import makeMappingArray
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

    def histogram_map_sum(self):

        name_list = []

        for i in range(10):
            name_list.append("map_"+str(i))

        tmp_real = self.datasets_fbe.return_realarea()
        tmp_area_fbe, tmp_path_fbe = self.datasets_fbe.get_sum()
        tmp_area_i8f, tmp_path_i8f = self.datasets_8if.get_sum()
        tmp_area_i8ff, tmp_path_i8ff = self.datasets_8iff.get_sum()
        tmp_area_i4f, tmp_path_i4f = self.datasets_4if.get_sum()
        tmp_area_i4ff, tmp_path_i4ff = self.datasets_4iff.get_sum()

        total_width, n =  0.8, 6
        width = total_width/n

        plt.xlabel('MapsID')
        plt.ylabel('Totoal areas of 10 times exploration(m^2)')
        x = list(range(len(tmp_real)))
        print("x:", x)
        plt.bar(x, tmp_real, width, label="Total_area", color='black')

        for i in range(len(x)):
            x[i] = x[i] + width
        print(x)
        plt.bar(x, tmp_area_fbe, width, label="fbe", color='blue')
        
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_area_i8f, width, label="8if", color='gray')

        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_area_i8ff, width, label="8iff", color='red', tick_label=name_list)

        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_area_i4f, width, label="4if", color='pink')
        
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_area_i4ff, width, label="4iff", color='green')

        # plt.ylim(0, max(data_list))     # y轴取值范围
        # plt.ylabel("TotalPath/TotalArea")
 
        # plt.xticks([index + 0.2 for index in x], label_list)
        # plt.xlabel("TestMaps")
        # plt.title("Cost of length path(m) per unit area(m^2)")
        plt.legend()
        plt.show()

    def histogram_map_median(self):

        name_list = []

        for i in range(10):
            name_list.append("map_"+str(i))

        tmp_real = self.datasets_fbe.return_realarea()
        for i in range(len(tmp_real)):
            tmp_real[i] = tmp_real[i]*0.1
        tmp_area_fbe, tmp_path_fbe = self.datasets_fbe.get_median()
        tmp_area_i8f, tmp_path_i8f = self.datasets_8if.get_median()
        tmp_area_i8ff, tmp_path_i8ff = self.datasets_8iff.get_median()
        tmp_area_i4f, tmp_path_i4f = self.datasets_4if.get_median()
        tmp_area_i4ff, tmp_path_i4ff = self.datasets_4iff.get_median()

        total_width, n =  0.8, 6
        width = total_width/n

        plt.xlabel('MapsID')
        plt.ylabel('mean areas of 10 times exploration(m^2)')
        x = list(range(len(tmp_real)))
        print("x:", x)
        plt.bar(x, tmp_real, width, label="Total_area", color='black')

        for i in range(len(x)):
            x[i] = x[i] + width
        print(x)
        plt.bar(x, tmp_area_fbe, width, label="fbe", color='blue')
        
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_area_i8f, width, label="8if", color='gray')

        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_area_i8ff, width, label="8iff", color='red', tick_label=name_list)

        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_area_i4f, width, label="4if", color='pink')
        
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_area_i4ff, width, label="4iff", color='green')

        # plt.ylim(0, max(data_list))     # y轴取值范围
        # plt.ylabel("TotalPath/TotalArea")
 
        # plt.xticks([index + 0.2 for index in x], label_list)
        # plt.xlabel("TestMaps")
        # plt.title("Cost of length path(m) per unit area(m^2)")
        plt.legend()
        plt.show()

    def histogram_path_sum(self):

        name_list = []

        for i in range(10):
            name_list.append("map_"+str(i))

        tmp_real = self.datasets_fbe.return_realarea()
        tmp_area_fbe, tmp_path_fbe = self.datasets_fbe.get_sum()
        tmp_area_i8f, tmp_path_i8f = self.datasets_8if.get_sum()
        tmp_area_i8ff, tmp_path_i8ff = self.datasets_8iff.get_sum()
        tmp_area_i4f, tmp_path_i4f = self.datasets_4if.get_sum()
        tmp_area_i4ff, tmp_path_i4ff = self.datasets_4iff.get_sum()

        total_width, n =  0.8, 6
        width = total_width/n

        plt.xlabel('MapsID')
        plt.ylabel('Totoal length of path of 10 times exploration(m)')
        x = list(range(len(tmp_path_fbe)))
        plt.bar(x, tmp_path_fbe, width, label="fbe", color='blue')
        
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_path_i8f, width, label="8if", color='gray')

        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_path_i8ff, width, label="8iff", color='red', tick_label=name_list)

        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_path_i4f, width, label="4if", color='pink')

        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_path_i4ff, width, label="4iff", color='green')

        # plt.ylim(0, max(data_list))     # y轴取值范围
        # plt.ylabel("TotalPath/TotalArea")
 
        # plt.xticks([index + 0.2 for index in x], label_list)
        # plt.xlabel("TestMaps")
        # plt.title("Cost of length path(m) per unit area(m^2)")
        plt.legend()
        plt.show()

    def histogram_path_median(self):
    
        name_list = []

        for i in range(10):
            name_list.append("map_"+str(i))

        tmp_real = self.datasets_fbe.return_realarea()
        tmp_area_fbe, tmp_path_fbe = self.datasets_fbe.get_median()
        tmp_area_i8f, tmp_path_i8f = self.datasets_8if.get_median()
        tmp_area_i8ff, tmp_path_i8ff = self.datasets_8iff.get_median()
        tmp_area_i4f, tmp_path_i4f = self.datasets_4if.get_median()
        tmp_area_i4ff, tmp_path_i4ff = self.datasets_4iff.get_median()

        total_width, n =  0.8, 6
        width = total_width/n

        plt.xlabel('MapsID')
        plt.ylabel('mean length of path of 10 times exploration(m)')
        x = list(range(len(tmp_path_fbe)))
        plt.bar(x, tmp_path_fbe, width, label="fbe", color='blue')
        
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_path_i8f, width, label="8if", color='gray')

        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_path_i8ff, width, label="8iff", color='red', tick_label=name_list)

        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_path_i4f, width, label="4if", color='pink')

        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, tmp_path_i4ff, width, label="4iff", color='green')

        # plt.ylim(0, max(data_list))     # y轴取值范围
        # plt.ylabel("TotalPath/TotalArea")
 
        # plt.xticks([index + 0.2 for index in x], label_list)
        # plt.xlabel("TestMaps")
        # plt.title("Cost of length path(m) per unit area(m^2)")
        plt.legend()
        plt.show()

    def meanArea_10maps(self):
        tmp_real = self.datasets_fbe.return_realarea()
        tmp_area_fbe, tmp_path_fbe = self.datasets_fbe.get_sum()
        tmp_area_i8f, tmp_path_i8f = self.datasets_8if.get_sum()
        tmp_area_i8ff, tmp_path_i8ff = self.datasets_8iff.get_sum()
        tmp_area_i4f, tmp_path_i4f = self.datasets_4if.get_sum()
        tmp_area_i4ff, tmp_path_i4ff = self.datasets_4iff.get_sum()
        
        meanMap_real = np.sum(tmp_real)
        meanMap_fbe = np.sum(tmp_area_fbe)   # [num]
        meanMap_i8f = np.sum(tmp_area_i8f)   # [num]
        meanMap_i8ff = np.sum(tmp_area_i8ff)   # [num]
        meanMap_i4f = np.sum(tmp_area_i4f)   # [num]
        meanMap_i4ff = np.sum(tmp_area_i4ff)   # [num]
        
        meanPath_fbe = np.sum(tmp_path_fbe)   # [num]
        meanPath_i8f = np.sum(tmp_path_i8f)   # [num]
        meanPath_i8ff = np.sum(tmp_path_i8ff)   # [num]
        meanPath_i4f = np.sum(tmp_path_i4f)   # [num]
        meanPath_i4ff = np.sum(tmp_path_i4ff)   # [num]
        
        print("=================== Mean Map(m^2) =======================")
        print("=====> meanMap_fbe: {:.2f}".format(meanMap_fbe/100))
        print("=====> meanMap_i4f: {:.2f}".format(meanMap_i4f/100))
        print("=====> meanMap_i4ff: {:.2f}".format(meanMap_i4ff/100))
        print("=====> meanMap_i8f: {:.2f}".format(meanMap_i8f/100))
        print("=====> meanMap_i8ff: {:.2f}".format(meanMap_i8ff/100))
        print("====================================================")
        
        print("\n=================== Mean path(m) =======================")
        print("=====> meanPath_fbe: {:.2f}".format(meanPath_fbe/100))
        print("=====> meanPath_i4f: {:.2f}".format(meanPath_i4f/100))
        print("=====> meanPath_i4ff: {:.2f}".format(meanPath_i4ff/100))
        print("=====> meanPath_i8f: {:.2f}".format(meanPath_i8f/100))
        print("=====> meanPath_i8ff: {:.2f}".format(meanPath_i8ff/100))
        print("====================================================")
        
Visualization_data().meanArea_10maps()