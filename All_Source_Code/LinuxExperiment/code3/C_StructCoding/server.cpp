#include <netinet/in.h> 
#include <arpa/inet.h>
#include <memory.h>
#include <unistd.h>  
#include <stdio.h> 
#include <iostream>

typedef void (* TCPServer)(int nConnectedSocket, int nListenSocket);
const int buf_size = 1024;

//该函数负责监听套接字，绑定端口，监听，与客户端建立连接。nlengthOfQueueOflisten 监听队列长度
int RunTCPServer(TCPServer ServerFunction, int nPort, int nLengthOfQueueOfListen = 100, const char *strBoundIP = NULL)
{
    /*
        AF_INET代表TCP/IP协议族，定义于数据结构sockaddr_in中
        ::socket返回一个套接口描述符，后续需要将这个描述符绑定在系统的一个端口上
    */
    int nListenSocket = ::socket(AF_INET, SOCK_STREAM, 0);
    if(-1 == nListenSocket)
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

    // 静态绑定IP，如果没有，则为本地IP
    if(NULL == strBoundIP)
    {
        /*
            sin_addr.s_addr是sockaddr_in结构中按网络字节顺序存储IP地址
            htonl()的作用是把主机自己顺序转化为网络字节顺序
        */
        ServerAddress.sin_addr.s_addr = htonl(INADDR_ANY);
    }
    else
    {
        // 如果有，则绑定。inet_pton将参数2由十进制转化为二进制，存在第三个参数里
        if(::inet_pton(AF_INET, strBoundIP, &ServerAddress.sin_addr) != 1)
        {
            std::cout << "inet_pton error" << std::endl;
            ::close(nListenSocket);
            return -1;
        }
    }

    // 绑定端口，主机字节顺序转换为网络字节顺序
    ServerAddress.sin_port = htons(nPort);
    // 套接字与ip和端口号的绑定
    if(::bind(nListenSocket, (sockaddr *)&ServerAddress, sizeof(sockaddr_in)) == -1)
    {
        std::cout << "Error during bind:"<<strerror(errno)<< std::endl;
        ::close(nListenSocket);
        return -1;
    }

    // 监听来自客户端的连接请求，参数1是套接字，参数2是监听队列长度的上限值
    if(::listen(nListenSocket, nLengthOfQueueOfListen) == -1)
    {
        std::cout << "listen error" << std::endl;
        ::close(nListenSocket);
        return -1;
    }
    
    sockaddr_in ClientAddress;
    socklen_t LengthOfClientAddress = sizeof(sockaddr_in); 
    
    // accept()接受连接
    int nConnectedSocket = ::accept(nListenSocket, (sockaddr *)&ClientAddress, &LengthOfClientAddress);
    if(-1 == nConnectedSocket)
    {
        std::cout << "accept error" << std::endl;
        ::close(nListenSocket);
        return -1;
    }
    
    ServerFunction(nConnectedSocket, nListenSocket);

    ::close(nConnectedSocket);
    ::close(nListenSocket);

    return 0;
}

void MyServer(int nConnectedSocket, int nListenSocket)
{
    int str_len,recv_len, recv_cnt;
    char message[buf_size];
    std::cout<<"Start accepting........"<<std::endl;
    while((str_len = read(nConnectedSocket, message, buf_size)) != 0){
        std::cout<<"Message from client :"<<message<<std::endl;
        write(nConnectedSocket, message, str_len);
    }
}

int main()
{  
    RunTCPServer(MyServer, 3845);
    return 0;
}