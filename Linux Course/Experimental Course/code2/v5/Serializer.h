#include "Temp.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>

using namespace std;

class Serializer{
    private:
        vector<ILSerializable*> m_vSerialized;
        
    public:
        // 多类多对象的序列化方法，参数中定义一个可变长数组
        bool Serialize(const char *pFilePath, std::vector<ILSerializable*>& v)
        {
            FILE* fp = fopen(pFilePath, "w+");
            if(fp == NULL) return false;
            for(int i = 0; i < v.size(); i++)
            {
                int type;
                v[i]->GetType(type);
                fwrite(&type, sizeof(int), 1, fp);
                v[i]->Serialize(fp);
            }
            fclose(fp);
            cout << "MutiTemp Serialize finish" << endl;
            return true;
        }

        // 多类多对象的反序列化方法，参数中定义一个可变长数组
        bool Deserialize(const char *pFilePath, std::vector<ILSerializable*>& v)
        {
            FILE* fp = fopen(pFilePath, "r+");
            if(fp == NULL) return false;
            for(;;)
            {
                int nType = -1;
                int r = fread(&nType, sizeof(int), 1, fp);

                int type;
                for(int i = 0; i < m_vSerialized.size(); i++)
                {
                    m_vSerialized[i]->GetType(type);
                    if(type == nType)
                    {
                        ILSerializable *p = m_vSerialized[i]->Deserialize(fp);
                        if(p != NULL) v.push_back(p);
                    }
                }
                if(r == 0) break;
            }
            fclose(fp);
            cout << "MutiTemp Deserialize finish" << endl;
            return true;
        }

        // 将ILSerializable类对象加入可变长数组
        void Register(ILSerializable *pSerialized)
        {
            m_vSerialized.push_back(pSerialized);
        }
};