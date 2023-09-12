#include <iostream>
#include "CLLogger.h"

using namespace std;

int main()
{
	CLLogger logger;
	logger.WriteLog("this is an error", 5);
	logger.WriteLog("another error", 6);

	return 0;
}