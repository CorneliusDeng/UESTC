#include <sys/socket.h>
#include <netinet/in.h>
#include <memory.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>

const int buf_size = 1024;

class TCP_Client
{
    private:
        int m_nServerPort;
        char *m_strServerIP;
        virtual void ClientFunction(int nClientSocket){
        }

    public:
        TCP_Client(int nServerPort, const char *strServerIP){
            m_nServerPort = nServerPort;

            int nlength = strlen(strServerIP); // 返回字符串长度
            m_strServerIP = new char[nlength + 1]; 
            strcpy(m_strServerIP, strServerIP); //复制字符串
        }
        virtual ~TCP_Client(){
            delete [] m_strServerIP; //删除
        }

        int Run(){
            // 创建套接字
            int nClientSocket = ::socket(AF_INET, SOCK_STREAM, 0);
            if(-1 == nClientSocket)
            {
                std::cout << "socket error" << std::endl;
                return -1;
            }

            sockaddr_in ServerAddress;
            memset(&ServerAddress, 0, sizeof(sockaddr_in)); //清空ServerAddress
            ServerAddress.sin_family = AF_INET;

            if(::inet_pton(AF_INET, m_strServerIP, &ServerAddress.sin_addr) != 1)
            {
                std::cout << "inet_pton error" << std::endl;
                ::close(nClientSocket);
                return -1;
            }

            // 绑定端口，主机字节顺序转换为网络字节顺序
            ServerAddress.sin_port = htons(m_nServerPort);
            
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
};

class My_TCP_Client : public TCP_Client
{
    public:
        My_TCP_Client(int nServerPort, const char *strServerIP) :TCP_Client(nServerPort, strServerIP)
        {

        }

        virtual ~My_TCP_Client()
        {

        }
    private:
        virtual void ClientFunction(int nClientSocket){
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
};

int main()
{
    My_TCP_Client myclient(3845, "127.0.0.1");
    myclient.Run();
    return 0;
}