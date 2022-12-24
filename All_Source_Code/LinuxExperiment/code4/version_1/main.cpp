#include <dlfcn.h>
#include <iostream>

using namespace std;

int main()
{
    /*
        打开动态链接库
        参数1是动态链接库的文件名，参数2是动态链接库的使用方法（RTLD_LAZY：动态地加入动态链接库中的函数）
        返回值：引动动态链接库的句柄，出错返回NULL
    */
    void *handle = dlopen("./libfunc.so", RTLD_LAZY);
    if(handle == 0)
    {
        cout << "dlopen error" << endl;
        return 0;
    }

    // 定义一个函数指针
    typedef void (*FUNC_PRINT)();
    
    /*
        映射动态链接库中的函数
        参数1是dlopen的返回值，参数2是动态链接库中的函数名
        返回值：“Print”被加载后在进程地址空间中的地址，出错返回NULL
    */ 
    FUNC_PRINT dl_print = (FUNC_PRINT)dlsym(handle, "Print");
    if(dl_print == 0)
    {
        cout << "dlsym error" << endl;
        return 0;
    }

    (dl_print)();

    // 卸载动态链接库
    dlclose(handle);

    return 0;
}