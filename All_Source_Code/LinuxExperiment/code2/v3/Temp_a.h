#include <stdio.h>
#include <iostream>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include <unistd.h>
using namespace std;

class Temp_a 
{
    private:
        int i;
    public:
        // 定义构造函数，实现构造函数重载
        Temp_a(){
            i  = 0;
        }
        Temp_a(int j){
            i = j;
        }
        void f(){
            cout<<"i = "<<i<<endl;
        }

        // 序列化方法
        bool Serialize(const char* pFilePath){
            // 打开文件,如果文件不存在,就创建文件
            int fd = open(pFilePath,O_RDWR | O_CREAT | O_TRUNC, 0);
            cout << "begin Serialize"<< endl; // 给个提示
            // 如果说打开文件或者创建文件错误,返回错误
            if(fd == -1){
                cout << "Serialize open error" << endl;
                return false;
            }
            // 如果说向文件里面写入数据时出错,出错原因有磁盘满,没有访问权限,超过了给定进程的文件长度等
            if (write(fd,&i,sizeof(int)) == -1)
            {
                cout << "Serialize write error" << endl;
                close(fd); // 关闭文件
                return false; // 返回错误
            }
            cout << "Serialize finish" << endl;
            // 如果上述打开与写都没有错误,那么则序列化成功
            return true; 
        }

        // 重载序列化方法
        bool Serialize(int fd) const
        {
            if(-1 == fd) return false;
            
            else if (write(fd,&i,sizeof(int)) == -1) return false;
            
            else return true;
        }

        // 反序列化方法
        bool Deserialize(const char* pFilePath){
            // 用读写的方式打开文件
            int fd = open(pFilePath,O_RDWR);
            cout << "Deserialize begin" << endl;
            // 打开文件错误
            if (fd == -1)
            {
                cout <<"Deserialize open error"<<endl;
                return false; // 返回错误
            }
            // 从序列化的文件读出数据
            int r = read(fd,&i,sizeof(int));
            if (r == -1)// 读文件出错
            {
                cout << "Deserialize read error";
                close(fd); // 关闭文件
                return false; // 返回错误
            }
            // 如果关闭文件错误
            if (close(fd) == -1){
            cout << "Deserialize close error" << endl;
                return false;// 返回错误
            }
            cout << "Deserialize finish" << endl;
            // 上述操作都成功,那么则反序列化成功
            return true;
        }
        
        // 重载反序列化方法
        bool Deserialize(int fd)
        {	
            if (-1 == fd) return false;
            int r = read(fd, &i, sizeof(int));
            if((0 == r) || (-1 == r)) return false;
            
            return true;
        } 
};