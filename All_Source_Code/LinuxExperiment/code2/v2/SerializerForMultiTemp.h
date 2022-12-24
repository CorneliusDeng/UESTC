#include "Temp.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>

using namespace std;

class SerializerForMultiTemp{

public:
    // 多个对象的序列化方法，参数中定义一个可变长数组
    static void Serialize(const char* pFilePath, vector<Temp *>& v){
        FILE* fp = fopen(pFilePath, "w+");
        for(int i = 0; i < v.size(); i++){
            fwrite(v[i], sizeof(int), 1, fp);
        }
        fclose(fp);
        cout << "MutiTemp Serialize finish" << endl;
    }

    // 多个对象的反序列化方法，参数中定义一个可变长数组
    static void Deserialize(const char* pFilePath, vector<Temp *>& v){
        FILE *fp = fopen(pFilePath, "r");
        for(int i = 0; i < v.size(); i++){
            fread(v[i], sizeof(int), 1, fp);
        }
        fclose(fp);
        cout << "MutiTemp Deserialize finish" << endl;
    }
};