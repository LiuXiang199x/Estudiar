import os
import sys

def lib_os():
    print(" =======> LIBS:os <======= ")
    print(os.getcwd())
    print(os.path.getsize("/home/agent/Estudiar/Python/libs_operation/lib_os.py"))
    print(os.path.isabs("/home/agent/Estudiar/Python/libs_operation/lib_os.py"))
    print(os.path.isfile("/home/agent/Estudiar/Python/libs_operation/lib_os.py"))
    print(os.path.isdir("/home/agent/Estudiar/Python/libs_operation/lib_os.py"))
    ret = os.listdir("/home/agent") # class - list
    print(ret)
    print(os.walk("/home/agent/Estudiar/OpenCV/Tutorial/"))
    """
    for i in range(131, 141):
        os.mknod("/home/agent/test_data2/"+str(i)+"/map.txt")
        os.mknod("/home/agent/test_data2/"+str(i)+"/path.txt")
    """ 
    print(os.path.split("/home/agent/test_data2/131/path.txt"))
    print(os.path.split("/home/agent/test_data2/131/path.txt")[1])
    print(os.path.splitext(os.path.split("/home/agent/test_data2/131/path.txt")[1]))
    
    """   
    os: This module provides a portable way of using operating system dependent functionality.
    这个模块提供了一种方便的使用操作系统函数的方法。
    """
    """
    os.remove(‘path/filename’) 删除文件
    os.rename(oldname, newname) 重命名文件
    os.walk() 生成目录树下的所有文件名
    os.chdir('dirname') 改变目录
    os.mkdir/makedirs('dirname')创建目录/多层目录
    os.rmdir/removedirs('dirname') 删除目录/多层目录
    os.listdir('dirname') 列出指定目录的文件
    os.getcwd() 取得当前工作目录
    os.chmod() 改变目录权限
    os.path.basename(‘path/filename’) 去掉目录路径，返回文件名
    os.path.dirname(‘path/filename’) 去掉文件名，返回目录路径
    os.path.join(path1[,path2[,...]]) 将分离的各部分组合成一个路径名
    os.path.split('path') 返回( dirname(), basename())元组
    os.path.splitext() 返回 (filename, extension) 元组
    os.path.getatime\ctime\mtime 分别返回最近访问、创建、修改时间
    os.path.getsize() 返回文件大小
    os.path.exists() 是否存在
    os.path.isabs() 是否为绝对路径
    os.path.isdir() 是否为目录
    os.path.isfile() 是否为文件
    """
    
    
def lib_sys():
    print(" =======> LIBS:sys <======= ")
    print(sys.version)
    print(sys.path)
    # sys.argv 都是以 str形式输入的
    # sys.argv 是默认长度为1的，自带一个自己文件路径 ['lib_os_sys.py']
    for item in sys.argv:
        print("we are in sys.argv loop")
        print(len(sys.argv))
        print(sys.argv)
        print(item)
        print(type(item))
    """
    sys: This module provides access to some variables used or maintained by 
    the interpreter and to functions that interact strongly with the interpreter.
    这个模块可供访问由解释器使用或维护的变量和与解释器进行交互的函数。
    """
    """
    sys.argv 命令行参数List，第一个元素是程序本身路径
    sys.modules.keys() 返回所有已经导入的模块列表
    sys.exc_info() 获取当前正在处理的异常类,exc_type、exc_value、exc_traceback当前处理的异常详细信息
    sys.exit(n) 退出程序，正常退出时exit(0)
    sys.hexversion 获取Python解释程序的版本值，16进制格式如：0x020403F0
    sys.version 获取Python解释程序的版本信息
    sys.maxint 最大的Int值
    sys.maxunicode 最大的Unicode值
    sys.modules 返回系统导入的模块字段，key是模块名，value是模块
    sys.path 返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值
    sys.platform 返回操作系统平台名称
    sys.stdout 标准输出
    sys.stdin 标准输入
    sys.stderr 错误输出
    sys.exc_clear() 用来清除当前线程所出现的当前的或最近的错误信息
    sys.exec_prefix 返回平台独立的python文件安装的位置
    sys.byteorder 本地字节规则的指示器，big-endian平台的值是'big',little-endian平台的值是'little'
    sys.copyright 记录python版权相关的东西
    sys.api_version 解释器的C的API版本
    """

if __name__ == "__main__":
    # lib_os()
    lib_sys()
    