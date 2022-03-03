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
    path = '/media/agent/eb0d0016-e15f-4a25-8c28-0ad31789f3cb/scene_dataset/data/train'
    path_val = '/media/agent/eb0d0016-e15f-4a25-8c28-0ad31789f3cb/scene_dataset/data/val'
# save_path = sys.argv[2]
    txt = '/media/agent/eb0d0016-e15f-4a25-8c28-0ad31789f3cb/Scene/DEDUCE/download_bedroom.txt'

    file = '/media/agent/eb0d0016-e15f-4a25-8c28-0ad31789f3cb/Scene/DEDUCE/download_classes.txt'
    classes_2 = list()
    with open(file) as class_file:
        for line in class_file:
            cur_sn = line.split()[0]
            classes_2.append(cur_sn)
    classes_2 = tuple(classes_2)

    with open(txt, 'r') as f:
        for line in f.readlines():
            label=line.split()[1]
            img_name = line.split()[0].rsplit('/')[-1].strip()
            sn_name = line.split()[0].rsplit('/')[-1].strip().rsplit('_')[0]+'_'+line.split()[0].rsplit('/')[-1].strip().rsplit('_')[1]
            save_path = os.path.join(path, label)
            print(save_path)
            save_path_val = os.path.join(path_val, label)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(save_path_val):
                os.makedirs(save_path_val)
            if not sn_name in classes_2[200:]:
                thread = DownloadThread(download, (save_path+os.sep+img_name, line.split()[0]))
                thread.start()
                thread.join()
            if sn_name in classes_2[200:]:
                thread = DownloadThread(download, (save_path_val+os.sep+img_name, line.split()[0]))
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