#include <iostream>
#include <unistd.h>
#include "CLLogger.h"

using namespace std;

int  main(void)
{
	CLLogger::WriteLogMsg("Hello World!", 0);

	pid_t pid = fork();
	if(pid == -1)
	{
		cout << "fork error" << endl;
		return 0;
	}
	else if(pid == 0)
	{
		cout << "child id : " << getpid() << endl;
	}
	else
	{
		sleep(2);
		cout << "father id : " << getpid() << endl;
	}
	
	return 0;
}
