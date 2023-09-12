#include"Table.h"
#include"BinarySearchTree.h"
#include <iomanip>
#include <utility>
#include <vector>

Table* Table::table = nullptr;//表的实例初始化

Table::Table(){

    //flags：可读可写，文件添加到末尾，文件不存在则添加；
    srand(time(0));
    File = open(TableName, O_RDWR|O_APPEND|O_CREAT,S_IRUSR | S_IWUSR);
    if(-1==File){
        std::cout << "文件打开失败!" << "\n";
        exit(1);
    }
    
    //把文件内容读取到内存中
    int CurPos = lseek(File, 0, SEEK_END);
    records.clear();
    std::cout << CurPos <<"\n";
    if(CurPos==0){
        std::cout<<"------------------------------"<<"\n";
        std::cout<<"开始初始化数据库表"<<"\n";
        InitialRecords();
        std::cout<<"初始化数据库表成功"<<"\n";
        std::cout<<"------------------------------"<<"\n\n";
    }else{
        std::cout<<"------------------------------"<<"\n";
        std::cout<<"数据库文件已存在，正在读取数据库文件到内存"<<"\n";
        Record record;

        int RecordByte = sizeof(record.primary_key) + record.col.size()*sizeof(LwjType);
        int num = CurPos / RecordByte;
        #ifdef DEBUG
            std::cout<<"num:" << num  << " " <<RecordByte <<"\n";
        #endif
        if (lseek(File, 0, SEEK_SET)== -1)
        {
            std::cout<<"设置lseek为0失败" << "\n";
            exit(1);
        }
        #ifdef DEBUG
            std::cout<<"num:" << num  << " " << RecordByte <<"\n";
        #endif
        for(int i = 0;i < num;i++){
            int f = read(File,&record.primary_key,sizeof(record.primary_key));
            if(-1==f){
                std::cout<<"从数据库文件中读取记录失败"<<"\n";
                exit(1);
            }
            for(int j = 0;j < record.col.size();j++){
                read(File,&record.col[j],sizeof(record.col[j]));
                if(-1==f){
                    std::cout<<"从数据库文件中读取记录失败"<<"\n";
                    exit(1);
                }
            }
            //std::cout<<"num2:" << record.col[3] <<"\n";
            ShowRecord(record);
            //std::cout<<"num1:" << num  << " " << sizeof(Record) <<"\n";
            records.push_back(record);
        }
        std::cout<<"读取数据库文件到内存成功"<<"\n";
        std::cout<<"读取记录成功，已读取" << num <<" 个记录\n";
        std::cout<<"------------------------------"<<"\n\n";

        //设置lseek为END
        if (lseek(File, SEEK_END, SEEK_SET)== -1)
        {
            std::cout<<"设置lseek为END失败" << "\n";
            exit(1);
        }   
    }
    
    //初始化索引文件

    //待写
}

Table::~Table(){
    if(-1!=File||0!=File){
        close(File);
    }
    records.clear();
}

//单例模式，用于只创建一个实例
Table* Table::GetTable(){
    if(table==nullptr){
        table = new Table();
    }
    return table;
}

//初始化整个表格
void Table::InitialRecords(){
    Record record;
    for(int i = 1;i <= InitRecordNum;i++){
        record = CreateRecord();
        //ShowRecord(record);
        //records.push_back(record);
        int f = SaveRecord(record);
        if(-1==f){
            std::cout<<"初始化文件保存记录失败"<<"\n";
            exit(1);
        }
    }
    std::cout<<"初始化表成功!\n";
}

//添加记录
bool Table::InsertRecord(){
    //p

    if(records.size()==MaxRecordNum){
        std::cout<<"已达数据库最大容量"<<"\n";
        return false;
    }
    Record record = CreateRecord();
    std::cout<<"------------------------------"<<"\n";

    bool f = SaveRecord(record);
    if(!f){
        std::cout<<"添加记录到文件失败"<<"\n";
        exit(1);
    }
    std::cout<<"已成功添加一条记录在数据库"<<"\n";
    ShowRecord(record);
    //更新索引
    //v
    std::cout<<"------------------------------"<<"\n";
    return f;
} 

//搜索记录
bool Table::SearchRecord(int col,int left,int right){

    std::cout<<"------------------------------"<<"\n";
    std::cout<<"开始搜索数据库表"<<"\n";
    
    BSTtree *BST = new BSTtree();
    //首先判断是否有索引文件
    if (Is_Index_File_Exists(col)) {
        std::cout << "已查找到索引文件，正在读取..." << "\n";
        bool f = BST->RealReadFile(col);
        //BST->Realprint();
        if (!f) {
            std::cout << "读取索引文件失败" << "\n";
            return f;
        }
        std::cout << "读取索引文件成功" << "\n";
    } else {
        std::cout << "未查找到索引文件，正在创建..." << std::endl;
        //创建索引文件
        CreateIndex(col);
        BST->RealReadFile(col);
        std::cout << "已为第" << col << "列创建索引文件" << std::endl;
    }


    std::vector<LwjType> ans = BST->Realsearch(left,right);

    std::cout<<"查找成功\n"<<"\n";

    std::cout<<"搜索第 "<<col <<" 列区间 [" <<left <<","<<right<<"]的结果如下:"<<"\n";
    std::cout<<"显示top " <<fmin((int)ans.size(),MaxShowRecordNum)<<" 条记录\n";
    sort(ans.begin(),ans.end());
    int num = 0;
    for(int i = 0;i < fmin(ans.size(),MaxShowRecordNum);i++){
        num++;
        ShowRecord(records[ans[i]]);
    }
    if(!num){
        std::cout<<"未找到第 "<<col <<" 列 [" <<left <<","<<right<<"]的记录"<<"\n";
    }
    std::cout<<"------------------------------"<<"\n\n";
    return true;
}   

//随机创建记录
Record Table::CreateRecord(){
    
    Record record;
    record.primary_key = records.size() + 1;
    std::cout<<"------------------------------"<<"\n";
    std::cout<<"开始随机创建记录"<< "\n";
    for(int i = 0;i < record.col.size();i++){
        LwjType x = rand() % 20000;
        record.col[i] = x;
        //srand(rand()%20000);
    }
    ShowRecord(record);
    std::cout<<"随机创建记录成功"<< "\n";
    std::cout<<"------------------------------"<<"\n\n";
    records.push_back(record);
    return record;
}


//判断索引文件是否存在
bool Table::Is_Index_File_Exists(int col){
    char index_file_path[200];
    sprintf(index_file_path, "%s%d", INDEX_FILE_PATH, col);
    return (access(index_file_path, F_OK) == 0);
}

//保存一条记录
bool Table::SaveRecord(const Record &record){
    int f = write(File,&record.primary_key,sizeof(record.primary_key));
    if(-1==f){
        return false;
    }
    for(int j = 0;j < record.col.size();j++){
        f = write(File,&record.col[j],sizeof(record.col[j]));
        if(-1==f){
            return false;
        }
    }

    return true;
} 

//显示某一条记录
void Table::ShowRecord(const Record &record){
    std::cout << std::setiosflags(std::ios::left) << std::setw(7); 
    std::cout << record.primary_key << " ";
    for(int i = 0;i < record.col.size();i++){
        std::cout << std::setiosflags(std::ios::left) << std::setw(7); 
         std::cout << record.col[i] << " ";
    }
    std::cout << "\n";
}

bool Table::CreateIndex(int col){
    BSTtree *BST = new BSTtree();
    for(int i = 0;i < records.size();i++){
        std::pair<LwjType,LwjType>data = std::make_pair(records[i].col[col-1],i);
        BST->RealInsertNode(data,col);
    }
    std::cout<<"已为第 "<<col<<" 列创建索引\n";
    return true;
}