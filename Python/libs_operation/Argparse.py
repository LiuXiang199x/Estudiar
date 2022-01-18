# 1引入模块
import argparse

print("命令行解析")
print("Convert argument strings to objects and assign them as attributes of the namespace. \
    Return the populated namespace.")
print("parse_args()是将之前add_argument()定义的参数进行赋值，并返回相关的namespace。")
print("====================================\n")


def tutorial():
    # 2建立解析对象
    parser = argparse.ArgumentParser()
    
    # 3增加属性：给xx实例增加一个aa属性 # xx.add_argument("aa")
    parser.add_argument("echo")         
    
    # 4属性给与args实例： 把parser中设置的所有"add_argument"给返回到args子类实例当中
    # 那么parser中增加的属性内容都会在args实例中，使用即可。
    args = parser.parse_args()
    parser.parse_args()
    
    # 打印定位参数echo
    print(args.echo)

def test():
    parsers = argparse.ArgumentParser()
    parsers.add_argument("echo")
    args = parsers.parse_args()
    parsers.parse_args()
    