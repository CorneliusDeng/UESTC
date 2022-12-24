#include <netinet/in.h> 
#include <arpa/inet.h>
#include <memory.h>
#include <unistd.h>  
#include <stdio.h> 
#include <iostream>

const int buf_size = 1024;

// TCP封装
class TCP_Server{
    private:
        int m_nServerPort;
        char *m_strBoundIP;
        int m_nLengthOfQueueOfListen;

        virtual void ServerFunction(int nConnectedSocket,int nListenSocket){

        }

    public:
        TCP_Server(int nServerPort, int nLengthOfQueueOfListen = 100, const char *strBoundIP = NULL){
            m_nServerPort = nServerPort; // 端口
            m_nLengthOfQueueOfListen = nLengthOfQueueOfListen; // 长度

            if (NULL == strBoundIP)
                m_strBoundIP = NULL;
            else{
                int length = strlen(strBoundIP); //返回字符串长度
                m_strBoundIP = new char[length + 1];
                memcpy(m_strBoundIP, strBoundIP, length + 1); //复制内存
            }
        }
        virtual ~TCP_Server(){
            if(m_strBoundIP != NULL)
                delete [] m_strBoundIP;
        }

        int Run(){
            int nListenSocket = ::socket(AF_INET, SOCK_STREAM, 0); // 创建套接字
            if(-1 == nListenSocket)
            {
                std::cout << "socket error" << std::endl;
                return -1;
            }

            sockaddr_in ServerAddress;
            memset(&ServerAddress, 0, sizeof(sockaddr_in)); //清空ServerAddress
            ServerAddress.sin_family = AF_INET;

            if(NULL == m_strBoundIP)
            {
                ServerAddress.sin_addr.s_addr = htonl(INADDR_ANY); // 将主机字节顺序转化为网络字节顺序
            }
            else
            {
                if(::inet_pton(AF_INET, m_strBoundIP, &ServerAddress.sin_addr) != 1)
                {
                    std::cout << "inet_pton error" << std::endl;
                    ::close(nListenSocket);
                    return -1;
                }
            }

            ServerAddress.sin_port = htons(m_nServerPort);
            // 绑定套接字到计算机端口
            if(::bind(nListenSocket, (sockaddr *)&ServerAddress, sizeof(sockaddr_in)) == -1)
            {
                std::cout << "Error during bind:"<<strerror(errno)<< std::endl;
                ::close(nListenSocket);
                return -1;
            }

            // 监听来自客户端的连接请求，参数1是套接字，参数2是监听队列长度的上限值
            if(::listen(nListenSocket, m_nLengthOfQueueOfListen) == -1)
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
};

class ServerForClient : public TCP_Server
{
    public:
        ServerForClient(int nServerPort, int nLengthOfQueueOfListen = 100, const char *strBoundIP = NULL): TCP_Server(nServerPort){
        }
        
        virtual ~ServerForClient(){
        }

    private:
        // 服务器类，向B校学生提供服务的具体实现
        virtual void ServerFunction(int nConnectedSocket, int nListenSocket){
            
        }
};

int main()
{
    ServerForClient myserver(3945);
    myserver.Run();
    return 0;
}