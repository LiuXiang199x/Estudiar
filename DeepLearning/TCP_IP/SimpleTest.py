import paramiko

class ConnectShell:
 
    def remotConnect(self):
        # 服务器相关信息,下面输入你个人的用户名、密码、ip等信息
        ip = "10.110.1.252"
        port = 22
        user = "liuxiang"
        password = "liuxiang"
 
        # 创建SSHClient 实例对象
        ssh = paramiko.SSHClient()
        # 调用方法，表示没有存储远程机器的公钥，允许访问
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # 连接远程机器，地址，端口，用户名密码
        ssh.connect(ip, port, user, password, timeout=10)
        # 输入linux命令
        ls = "ls"
        stdin, stdout, stderr = ssh.exec_command(ls)
        # 输出命令执行结果
        result = stdout.read()
        print("result:", result)

        ls = "pwd"
        stdin, stdout, stderr = ssh.exec_command(ls)
        # 输出命令执行结果
        result = stdout.read()
        print("result:", result)
        
        # 关闭连接
        ssh.close()
        
ConnectShell().remotConnect()