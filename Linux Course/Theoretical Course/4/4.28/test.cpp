#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include "LibExecutive.h"
#include "CLProcess.h"
#include "CLProcessFunctionForExec.h"

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

		CLExecutive *process = new CLProcess(new CLProcessFunctionForExec, true);
		
		{
			CLMutex mutex("TestForSharedMutex", MUTEX_USE_SHARED_PTHREAD);
			CLCriticalSection cs(&mutex);

			if((process->Run((void *)"./test/a.out")).IsSuccess())
			{
				cout << "in parent" << endl;
				sleep(5);
			}
			else
			{
				cout << "process Run error" << endl;
				throw CLStatus(-1, 0);
			}
		}

		process->WaitForDeath();

		throw CLStatus(0, 0);
	}
	catch(CLStatus& s)
	{
		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}