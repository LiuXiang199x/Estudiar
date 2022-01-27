# ！/usr/bin/env python
# -*- coding:utf-8 -*-
import os

from urllib3 import *
import threading
import os
http = PoolManager()
disable_warnings()  # 禁用警告

# url_data = len(urllist)
#多线程类
class DownloadThread(threading.Thread):
    def __init__(self, func, args):
        super().__init__(target=func, args=args)

def download(filename, url):
    response = http.request('GET', url)
    f = open(filename,'wb')  # wb的b表示我们要写的文件是一个二进制的文件
    f.write(response.data)
    f.close()
    print('<',filename,'>','下载完成。')

if __name__ == '__main__':
    path = '/home/agent/Estudiar/Python/downloadhttp.py/pics'
# save_path = sys.argv[2]
    txt = '/home/agent/Estudiar/Python/downloadhttp.py/test_toiletroom.txt'

    with open(txt, 'r') as f:
        for line in f.readlines():
            label=line.split()[1]
            img_name = line.split()[0].rsplit('/')[-1].strip().rsplit('_')[2]
            sn_name = line.split()[0].rsplit('/')[-1].strip().rsplit('_')[0]+'_'+line.split()[0].rsplit('/')[-1].strip().rsplit('_')[1]
            save_path_label = os.path.join(path, label)
            save_path = os.path.join(save_path_label, sn_name)
            print(save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            thread = DownloadThread(download, (save_path+os.sep+img_name, line.split()[0]))
            thread.start()
            thread.join()



#
# path = r'C:\Users\97102\Desktop\新建文件夹'
# txt = r'C:\Users\97102\Downloads\task.txt'
# os.makedirs(path) if not os.path.exists(path) else False
# with open(txt, 'r') as f:
#     for line in f.readlines():
#         label=line.split()[1]
#         # print(line.split()[0])
#         img_name = line.split()[0].rsplit('/')[-1].strip()
#         t1 = threading.Thread(target=request.urlretrieve, args=(path+os.sep+label+os.sep+img_name, line.strip()[0],))
#         t1.start()
#         t1.join()