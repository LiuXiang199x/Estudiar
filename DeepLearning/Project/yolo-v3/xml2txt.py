import math
import xml.etree.cElementTree as et
import os

class_num={
    'person':0,
    'horse':1,
    'bicycle':2
}

xml_dir='data/image_voc'
xml_filenames=os.listdir(xml_dir)
with open('data.txt','a') as f:
    for xml_filename in xml_filenames:
        xml_filename_path=os.path.join(xml_dir,xml_filename)
        tree=et.parse(xml_filename_path)
        root=tree.getroot()
        filename=root.find('filename')
        names=root.findall('object/name')
        boxes=root.findall('object/bndbox')

        data=[]
        data.append(filename.text)
        for name,box in zip(names,boxes):
            cls=class_num[name.text]
            cx,cy,w,h=math.floor((int(box[2].text)-int(box[0].text))/2),math.floor((int(box[3].text)-int(box[1].text))/2),int(box[2].text)-int(box[0].text),int(box[3].text)-int(box[1].text)
            data.append(cls)
            data.append(cx)
            data.append(cy)
            data.append(w)
            data.append(h)
        _str=''
        for i in data:
            _str=_str+' '+str(i)
        f.write(_str+'\n')
f.close()