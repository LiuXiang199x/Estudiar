import numpy as np
import csv

file_pth = "/home/marco/下载/facebook-contest_export.csv"
f = open(file_pth)
f = csv.DictReader(f)
for row in f:
    print(row)