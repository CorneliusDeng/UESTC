#include <stdio.h>
#include <iostream>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include <unistd.h>
using namespace std;


//  定义ILSerializable类，将其方法都设置为虚函数，以实现多态
class ILSerializable
{
    public:
        virtual bool Serialize(FILE* fp) = 0;
        virtual ILSerializable* Deserialize(FILE* fp) = 0;
        virtual bool GetType(int& type) = 0;

        ILSerializable()
        {
        }

        virtual ~ILSerializable()
        {
        }
};

// Temp_a类继承ILSerializable类
class Temp_a : public ILSerializable
{
    private:
        int i;
    public:
        // 定义构造函数，实现构造函数重载
        Temp_a(){
            i = 0;
        }
        Temp_a(int j){
            i = j;
        }
        void f(){
            cout<<"打印a类: i = "<<i<<endl;
        }

        // 获取当前的类别
        virtual bool GetType(int& type){
            type = 0;
            return true;
        }

        // 序列化方法
        virtual bool Serialize(FILE *fp){
            if(fp == NULL) return false;
            fwrite(&i, sizeof(int), 1, fp);
            return true;
        }

        // 反序列化方法
        virtual ILSerializable* Deserialize(FILE *fp){
            Temp_a *p = new Temp_a();
            fread(&(p->i), sizeof(int), 1, fp);
            return p;
        }
};

// Temp_b类继承ILSerializable类
class Temp_b : public ILSerializable
{
    private:
        int i, j;
    public:
        // 定义构造函数，实现构造函数重载
        Temp_b(){
            i = 0;
            j = 0;
        }
        Temp_b(int k){
            i = k;
            j = k + 1;
        }
        void f(){
            cout<<"打印b类: i = "<<i<<", j = "<<j<<endl;
        }

        // 获取当前的类别
        virtual bool GetType(int& type){
            type = 1;
            return true;
        }

        // 序列化方法
        virtual bool Serialize(FILE *fp){
            if(fp == NULL) return false;
            fwrite(&i, sizeof(int), 1, fp);
            fwrite(&j, sizeof(int), 1, fp);
            return true;
        }

        // 反序列化方法
        virtual ILSerializable*  Deserialize(FILE* fp){
            Temp_b *p = new Temp_b();
            fread(&(p->i), sizeof(int), 1, fp);
            fread(&(p->j), sizeof(int), 1, fp);
            return p;
        }
};

// Temp_c类继承ILSerializable类
class Temp_c : public ILSerializable
{
    private:
        int i, j, k;
    public:
        // 定义构造函数，实现构造函数重载
        Temp_c(){
            i = 0;
            j = 0;
            k = 0;
        }
        Temp_c(int l){
            i = l;
            j = l + 1;
            k = l + 2;
        }
        void f(){
            cout<<"打印c类: i = "<<i<<", j = "<<j<<", k = "<<k<<endl;
        }

        // 获取当前的类别
        virtual bool GetType(int& type){
            type = 2;
            return true;
        }

        // 序列化方法
        virtual bool Serialize(FILE *fp){
            if(fp == NULL) return false;
            fwrite(&i, sizeof(int), 1, fp);
            fwrite(&j, sizeof(int), 1, fp);
            fwrite(&k, sizeof(int), 1, fp);
            return true;
        }

        // 反序列化方法
        virtual ILSerializable*  Deserialize(FILE* fp)
        {
            Temp_c *p = new Temp_c();
            fread(&(p->i), sizeof(int), 1, fp);
            fread(&(p->j), sizeof(int), 1, fp);
            fread(&(p->k), sizeof(int), 1, fp);
            return p;
        }
};