import os

from importlib_metadata import re
import torch


def save2txt():
    f = open('/home/agent/Estudiar/Python/test/test.txt','w')
    lisss = ["aaa", "bbb", "ccc"]

    for a in lisss:
        filename = a+"/"+"image"
        for i in range(9):
            input_txt = filename + " " + str(i) + "\n"
            f.write(input_txt)
    f.close()
    
def read2txt():
    f = open('/home/agent/Estudiar/Python/test/test.txt','r')
    for i in f:
        print(i)
        print(len(i))
    f.close()
    
a = torch.tensor(2)
print(a)
print(int(a))
print(a.int())