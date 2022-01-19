# 1引入模块
import argparse

print("命令行解析")
print("argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，\
    通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，\
        并自动生成帮助和使用信息。")
print("Convert argument strings to objects and assign them as attributes of the namespace. \
    Return the populated namespace.")
print("parse_args()是将之前add_argument()定义的参数进行赋值，并返回相关的namespace。")
print("====================================\n")


def tutorial():
    # 2建立解析对象
    parser = argparse.ArgumentParser()
    
    # 3增加属性：给xx实例增加一个aa属性 # xx.add_argument("aa")
    # parser.add_argument("echo",help=" echo the string you use here!")
    #向对象中添加相关命令行参数或选项,每一个add_argument方法对应一个参数或选项
    parser.add_argument("echo")         
    
    # 4属性给与args实例： 把parser中设置的所有"add_argument"给返回到args子类实例当中
    # 那么parser中增加的属性内容都会在args实例中，使用即可。
    
    #调用parse_args()方法进行解析
    args = parser.parse_args()
    parser.parse_args()
    
    # 打印定位参数echo
    print(args.echo)

def calc_sum_max():
    # 1. 创建一个解析器
    parsers = argparse.ArgumentParser(description="Desciptions: Calc max and sum value")
    
    # 2. 添加参数
    # 给ArgumentParser添加参数信息通过add_argument()方法完成
    # 这些调用指定 ArgumentParser 如何获取命令行字符串并将其转换为对象
    # 这些信息在 parse_args() 调用时被储存和使用
    parsers.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')
    parsers.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    # 调用 parse_args() 将返回一个具有 integers 和 accumulate 两个属性的对象。
    # integers 属性将是一个包含一个或多个整数的列表，
    # 而 accumulate 属性当命令行中指定了 --sum 参数时将是 sum() 函数，否则则是 max() 函数。


    # 3. 解析参数
    # 它将检查命令行，把每个参数转换为适当的类型然后调用相应的操作。
    # 在大多数情况下，这意味着一个简单的 Namespace 对象将从命令行解析出的属性构建
    args = parsers.parse_args()
    # parsers.parse_args()
    print(args.accumulate(args.integers))

def test_mine():
    parser = argparse.ArgumentParser(description="Description: My Test")
    # name要是 “XXX” 语句会按照顺序默认赋值. 直接python可以赋值：python X.py 1
    # 要是“-XX” “--XX”，那么 python X.py 1 是不行的，要指定name: python X.py -/--XX 1
    
    # parser.add_argument("add1", type=int, help="this is add1")
    # parser.add_argument("add2", type=float, help="this is add2")
    ## dest = "别名"
    # parser.add_argument("-qw", "-i", type=float, dest="v2")
    ## -qw 命令 和 -i 命令相同
    # print("add1:", args.add1)
    # print("add2:", args.add2)
    # print(args.add1 + args.v2)
    # print(args.add2 + args.v2)
    
    # 有 default 赋值的初始化变量是可以直接输出的
    parser.add_argument("--v4", type=int, default=22, help="default integer")
    args = parser.parse_args()
    print("v4:", args.v4)
    print(type(args.v4))

    parser.add_argument("--v5", type=int, 
                        nargs=3, help="default integer")
    args = parser.parse_args()
    print("v5:", args.v5)
    print(type(args.v5))

    # metavar 就是让显示的信息更加好看
 
def nargs_action():
    # ArgumentParser对象通常将一个动作与一个命令行参数关联。
    # nargs关键字参数将一个动作与不同数目的命令行参数关联在一起
    print("nargs: =\"N\", =\"*\"，=\"+\", =\"?\"")
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tolist', action='append')
    # python arg_action.py --tolist 1 --tolist 2 --tolist 3
    # output: ['1', '2', '3']

    parser.add_argument('--tolistt', action='append', nargs='+')
    args = parser.parse_args()
    print(args.tolistt)
    print(args.tolist)


nargs_action()