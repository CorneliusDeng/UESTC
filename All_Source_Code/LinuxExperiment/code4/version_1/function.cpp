#include <iostream>

using namespace std;

/* 
使用extern "C"使得导出函数名称和实际名称一致
告诉编译器按C语言的方式设定函数的导出名
*/
extern "C" void Print() 
{
    cout << "Hello China!" << endl;
}
