#include "Temp_a.h"
#include "Temp_b.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>

using namespace std;

// 定义序列化结构体
struct Serialized{
    int nType; // 对象所属类别
    void *pObj; // 实例化对象
};

class Serializer{
public:
    // 多类多对象的序列化方法，参数中定义一个可变长数组
    static bool Serialize(const char* pFilePath, vector<Serialized>& v){
        int fd = open(pFilePath,O_RDWR | O_CREAT | O_TRUNC, 0);
        // 如果说打开文件或者创建文件错误,返回错误
     	if(-1 == fd) return false;
     	for(int i = 0; i < v.size(); i++)
     	{
             // 如果说向文件里面写入数据时出错,出错原因有磁盘满,没有访问权限,超过了给定进程的文件长度等
            if(write(fd,&(v[i].nType),0) == -1)
            {
                close(fd);
                return false;
            }
            // 如果是Temp_a类别
            if(0 == v[i].nType)
            {
                Temp_a *p=(Temp_a *)(v[i].pObj);
                
                if(p->Serialize(fd)==false)
                    return false;
            }
            // 如果是Temp_b类别
            else if(1 == v[i].nType)
            {
                Temp_b *p=(Temp_b *)(v[i].pObj);
                
                if(p->Serialize(fd)==false)
                    return false;
            }
	    }
	    if(close(fd) == -1) return false;
        cout << "MutiTemp Serialize finish" << endl;
        return true;
    }

    // 多类多对象的反序列化方法，参数中定义一个可变长数组
    static bool Deserialize(const char* pFilePath, vector<Serialized>& v){
        int fd = open(pFilePath,O_RDWR | O_CREAT | O_TRUNC, 0);
     	if(-1 == fd) return false;
        
        // 无限循环
        for(;;)
      	{
      	   int nType;
      	   int r = read(fd, &nType, sizeof(Serialized));
           // 读结构体出错，或空数据
      	   if((-1 == r) || (0 == r)) 
                break;

      	   // 如果为类别Temp_a,将其写入可变长数组
      	   if(0 == nType)
      	   {
                Temp_a *p;
                p = new Temp_a();
                p->Deserialize(fd);
                Serialized s;
                s.nType = nType;
                s.pObj = p;
      	   	    v.push_back(s);
      	   }
           // 如果为类别Temp_b,将其写入可变长数组
      	   else if(1 == nType)
      	   {
      	   	    Temp_b *p;
      	   	    p = new Temp_b();
      	   	    p->Deserialize(fd);
      	   	    Serialized s;
      	   	    s.nType = nType;
      	   	    s.pObj = p;
      	   	    v.push_back(s);
      	   }
      	}
        if(close(fd) == -1) return false;
        cout << "MutiTemp Deserialize finish" << endl;
        return true;
    }
};