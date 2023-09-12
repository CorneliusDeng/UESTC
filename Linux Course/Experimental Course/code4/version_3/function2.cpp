#include <iostream>

using namespace std;

const int FUNC_ID = 2;

extern "C" void Print()
{
    cout << "Hello China!" << endl;
}

extern "C" void Help()
{
    cout << "Function ID " << FUNC_ID << " : This function will print hello china." << endl;
}

extern "C" int GetID()
{
    // 返回功能号
    return FUNC_ID;
}
