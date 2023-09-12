#include "Serializer.h"

int main()
{
    {
        // 实例化类别Temp_a、Temp_b、Temp_c的对象
        Temp_a a1(1);
        Temp_a a2(2);
        Temp_b b1(49);
        Temp_b b2(50);
        Temp_c c1(99);
        Temp_c c2(100);

        // 将对象加入数组
        vector<ILSerializable*> v;
        v.push_back(&a1);
        v.push_back(&a2);
        v.push_back(&b1);
        v.push_back(&b2);
        v.push_back(&c1);
        v.push_back(&c2);
        
        // 序列化数据到文件 
        Serializer s;
        s.Serialize("data", v);
    }

    {
        Serializer s;
        Temp_a a;
        Temp_b b;
        Temp_c c;
        s.Register(&a);
        s.Register(&b);
        s.Register(&c);

        // 从文件中反序列化
        vector<ILSerializable*> v;
        s.Deserialize("data", v);

        // 判断类别后打印当前反序列化的类别
        for(int i = 0; i < v.size(); i++)
        {
            Temp_a *p = dynamic_cast<Temp_a *>(v[i]);
            if(p != NULL) p->f();
            Temp_b *q = dynamic_cast<Temp_b *>(v[i]);
            if(q != NULL) q->f();
            Temp_c *r = dynamic_cast<Temp_c *>(v[i]);
            if(r != NULL) r->f();
        }
    }

    return 0;
}