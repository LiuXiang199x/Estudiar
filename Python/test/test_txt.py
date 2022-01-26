import os

f = open('/home/agent/Estudiar/Python/test/test.txt','w')
lisss = ["aaa", "bbb", "ccc"]

for a in lisss:
    filename = a+"/"+"image"
    for i in range(9):
        input_txt = filename + " " + str(i) + "\n"
        f.write(input_txt)
f.close()