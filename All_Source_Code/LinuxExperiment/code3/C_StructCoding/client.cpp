#include <sys/socket.h>
#include <netinet/in.h>
#include <memory.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>

typedef void (* TCPClient)(int nConnectedSocket);
const int buf_size = 1024;

int RunTCPClient(TCPClient ClientFunction, int nServerPort, const char *strServerIP)
{
    /*
        AF_INET代表TCP/IP协议族，定义于数据结构sockaddr_in中
        ::socket返回一个套接口描述符，后续需要将这个描述符绑定在系统的一个端口上
    */
    int nClientSocket = ::socket(AF_INET, SOCK_STREAM, 0);
    if(-1 == nClientSocket)
    {
        std::cout << "socket error" << std::endl;
        return -1;
    }

    // sockaddr_in是一种包含了地址族、网络数据端口号、IP地址等内容的数据结构
    sockaddr_in ServerAddress;
    // memset函数在socket中常用于清空一个结构类型变量或者数组，参数1是对应内存的起始地址，参数2将内存置为0，参数3为往后置为0的空间大小
    memset(&ServerAddress, 0, sizeof(sockaddr_in));
    // sin_family指的是协议族
    ServerAddress.sin_family = AF_INET;
    if(::inet_pton(AF_INET, strServerIP, &ServerAddress.sin_addr) != 1)
    {
        std::cout << "inet_pton error" << std::endl;
        ::close(nClientSocket);
        return -1;
    }

    // 绑定端口，主机字节顺序转换为网络字节顺序
    ServerAddress.sin_port = htons(nServerPort);
    
    // 判断是否连接服务器端
    if(::connect(nClientSocket, (sockaddr*)&ServerAddress, sizeof(ServerAddress)) == -1)
    {
        std::cout << "connect error" << std::endl;
        ::close(nClientSocket);
        return -1;
    }
    else    
        std::cout<<"Connect success"<<std::endl;
    
    ClientFunction(nClientSocket);
    ::close(nClientSocket);
    
    return 0;
}

void MyClient(int nClientSocket)
{
    char message[buf_size];
    int str_len;
    while(1)
    {
        fputs("Input message ('Q' or 'q' to quit) : ", stdout);
        fgets(message, buf_size, stdin);
        if(!strcmp(message, "q\n")|| !strcmp(message, "Q\n"))
            break;
        str_len = write(nClientSocket, message, strlen(message));
        
        // 接收从服务器端写入的数据
        int recv_len = 0, recv_cnt;
        while(recv_len < str_len)
        {
            recv_cnt = read(nClientSocket, &message[recv_len], buf_size - 1);
            if(recv_cnt == -1)
                std::cout<<"Read error"<<std::endl;
            recv_len += recv_cnt;
        }
        message[str_len] = 0;
        std::cout<<"Message from server :"<<message<<std::endl;
    }
}

int main()
{
    RunTCPClient(MyClient, 3845, "127.0.0.1");
    return 0;
}