#include <iostream>
#include "CLLogger.h"

using namespace std;

extern CLLogger g_logger;

class A
{
public:
    A()
    {
	CLStatus s = g_logger.WriteLog("in A()", 0);
	if(!s.IsSuccess())
	    cout << "g_logger.WriteLog error" << endl;
    }
};

A g_a;

CLLogger g_logger;

int main()
{
	return 0;
}
