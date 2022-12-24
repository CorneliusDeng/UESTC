#include <iostream>
#include <vector>
#include "SerializerForMultiTemp.h"

using namespace std;

int main(){
    // 实例化类Temp的两个对象a1，a2
    Temp a1(100);
    Temp a2(200);
    // 定义变长数组v1，元素类型都为类Temp的对象
    vector<Temp *> v1;
    // 将a1，a2放入变长数组v1
    v1.push_back(&a1);
    v1.push_back(&a2);
    // 序列化v1
    SerializerForMultiTemp::Serialize("data.txt", v1);

    // 实例化类Temp的两个对象a3，a4
    Temp a3;
    Temp a4;
    // 定义变长数组v2，元素类型都为类Temp的对象
    vector<Temp *> v2;
    // 将a3，a4放入变长数组v2
    v2.push_back(&a3);
    v2.push_back(&a4);
    // 反序列化v2
    SerializerForMultiTemp::Deserialize("data.txt", v2);
    a3.f();
    a4.f();
}