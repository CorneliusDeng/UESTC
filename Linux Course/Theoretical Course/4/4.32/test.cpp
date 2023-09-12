#include <iostream>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "LibExecutive.h"

using namespace std;

int main()
{
	try
	{
		if(!CLLibExecutiveInitializer::Initialize().IsSuccess())
		{
			cout << "Initialize error" << endl;
			return 0;
		}

		if((mkfifo("test_pipe", S_IRUSR | S_IWUSR) == -1) && (errno != EEXIST))
		{
			cout << "mkfifo error" << endl;
			throw CLStatus(-1, 0);
		}

		CLExecutive *process1 = new CLProcess(new CLProcessFunctionForExec, true);
		if(!((process1->Run((void *)"./child1/a.out")).IsSuccess()))
		{
			cout << "Run error" << endl;
			throw CLStatus(-1, 0);
		}

		CLExecutive *process2 = new CLProcess(new CLProcessFunctionForExec, true);
		if(!((process2->Run((void *)"./child2/a.out")).IsSuccess()))
		{
			cout << "Run error" << endl;
			process1->WaitForDeath();
			throw CLStatus(-1, 0);
		}

		int fd = open("test_pipe", O_RDONLY);
		if(fd != -1)
		{
			cout << "parent opened pipe" << endl;

			sleep(15);

			close(fd);
			cout << "parent closed pipe" << endl;
		}

		process1->WaitForDeath();
		process2->WaitForDeath();

		throw CLStatus(0, 0);
	}
	catch(CLStatus& s)
	{
		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}