#include <iostream>
#include "CLLogger.h"

using namespace std;

int main()
{
	CLLogger logger;
	CLStatus s = logger.WriteLog("this is an error", 5);
	if(!s.IsSuccess())
	{
		cout << "logger error" << endl;
	}
	else
	{
		cout << "logger success" << endl;
	}
}