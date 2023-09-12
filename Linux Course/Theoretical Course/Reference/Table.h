#ifndef TABLE_H
#define TABLE_H
#include<bits/stdc++.h>
#include <sys/types.h>//这里提供类型pid_t和size_t的定义
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define TableName "Table.bat"
#define LwjType int64_t
#define ColNum 10 //记录的属性个数
#define InitRecordNum 20//初始化记录个数
#define MaxRecordNum 1000000//数据库存放最大记录个数
#define MaxShowRecordNum 10//最多显示个数
//#define DEGUB

//用vector存放节点，避免节点重复
struct Element{
    LwjType key;//键值
    std::vector<LwjType> val; //重复的列号
};

/*
用于存储一条记录
大小固定的，方便存入文件
*/
struct Record{
    std::vector<LwjType> col = std::vector<LwjType>(ColNum,0);
    LwjType primary_key;
    //std::vector<LwjType> col =;
};


class Table{
private:
    
    std::vector<Record> records;//记录
    static Table *table;//表的实例
    int File;//表格文件标识
    static pthread_mutex_t *Mutex;//互斥的变量

    
public:
    Table();
    ~Table();
    static Table *GetTable();//单例模式，用于只创建一个实例
    bool InsertRecord();//添加记录
    bool SearchRecord(int col,int left,int right);//搜索记录
    Record CreateRecord();//随机创建记录
    bool SaveRecord(const Record &record); //保存一条记录
    void ShowRecord(const Record &record);//显示某一条记录
    void InitialRecords();//初始化整个表格
    bool Is_Index_File_Exists(int col);//判断索引文件是否存在
    bool CreateIndex(int col);//创建索引文件

};


#endif