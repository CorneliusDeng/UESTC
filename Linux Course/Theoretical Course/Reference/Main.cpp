#include "Table.h"

int main(){
    Table *table = Table::GetTable();
    //table->InitialRecords();
    //table->SearchRecord(1, 10, 2000);
    //table->InsertRecord();
    table->CreateIndex(9);
    //table->SearchRecord(2, 2, 2000);
    // std::cout<<sizeof(Record) << "\n";
    return 0;
}