#include "Serializer.h"

int main() {
	
	{
		// 实例化类别Temp_a的对象a1，实例结构体对象s1
		Temp_a a1(100);   
		Serialized s1;
		s1.nType = 0;
		s1.pObj = &a1;

		// 实例化类别Temp_a的对象a2，实例结构体对象s2
		Temp_a a2(99);  
		Serialized s2;	
		s2.nType = 0;
		s2.pObj = &a2;
		
		// 实例化类别Temp_b的对象b1，实例结构体对象s3
		Temp_b b1(1); 
		Serialized s3;
		s3.nType = 1;
		s3.pObj = &b1;
		
		// 实例化类别Temp_b的对象b2，实例结构体对象s4
		Temp_b b2(2);   
		Serialized s4;
		s4.nType = 1;
		s4.pObj = &b2;
		
		// 将结构体对象s1～s4放入数组
		vector<Serialized>v;
		v.push_back(s1);
		v.push_back(s2);
		v.push_back(s3);
		v.push_back(s4);
		
		// 序列化数据到文件
		Serializer s;
		s.Serialize("data",v);
	}
	
	{
		Serializer s;
		vector<Serialized>v;
		// 从文件中反序列化
		s.Deserialize("data",v);
		for(int i = 0; i < v.size(); i++)
		{
			if(v[i].nType == 0)
			{
				Temp_a *p=(Temp_a *)(v[i].pObj);
				p->f();
			}
			else if(v[i].nType == 1)
			{
				Temp_b *p = (Temp_b *)(v[i].pObj);
				p->f();
			}
		}
	}
	
	return 0;
}