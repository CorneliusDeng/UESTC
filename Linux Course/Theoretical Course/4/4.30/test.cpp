#include <iostream>
#include <pthread.h>
#include <unistd.h>
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

		CLSharedMemory smFlag("SharedFlag", sizeof(long));
		long *pFlag = (long *)smFlag.GetAddress();
		*pFlag = 0;

		CLConditionVariable cv("SharedCond");
		CLMutex mutex("SharedMutex", MUTEX_USE_SHARED_PTHREAD);

		CLExecutive *process = new CLProcess(new CLProcessFunctionForExec, true);
		if(!((process->Run((void *)"./test/a.out")).IsSuccess()))
		{
			cout << "Run error" << endl;
			throw CLStatus(-1, 0);
		}

		{
			CLCriticalSection cs(&mutex);

			while((*pFlag) == 0)
				if(!(cv.Wait(&mutex).IsSuccess()))
					cout << "Wait error" << endl;
		}

		cout << "in parent" << endl;

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