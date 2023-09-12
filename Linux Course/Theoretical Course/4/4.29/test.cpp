#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include "LibExecutive.h"

using namespace std;

int main()
{
	bool battr = false;
	pthread_condattr_t attr;

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

		if(pthread_condattr_init(&attr) != 0)
		{
			cout << "pthread_condattr_init error" << endl;
			throw CLStatus(-1, 0);
		}

		battr = true;

		if(pthread_condattr_setpshared(&attr, PTHREAD_PROCESS_SHARED) != 0)
		{
			cout << "pthread_condattr_setpshared error" << endl;
			throw CLStatus(-1, 0);
		}

		CLSharedMemory smCond("SharedCond", sizeof(pthread_cond_t));
		pthread_cond_t *pCond = (pthread_cond_t *)smCond.GetAddress();

		if(pthread_cond_init(pCond, &attr) != 0)
		{
			cout << "pthread_cond_init error" << endl;
			throw CLStatus(-1, 0);
		}

		CLConditionVariable cv(pCond);
		CLMutex mutex("SharedMutex", MUTEX_USE_SHARED_PTHREAD);

		CLExecutive *process = new CLProcess(new CLProcessFunctionForExec, true);

		if(!((process->Run((void *)"./test/a.out")).IsSuccess()))
		{
			cout << "Run error" << endl;

			if(pthread_cond_destroy(pCond) != 0)
				cout << "pthread_cond_destroy error" << endl;

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

		if(pthread_cond_destroy(pCond) != 0)
			cout << "pthread_cond_destroy error" << endl;

		throw CLStatus(0, 0);
	}
	catch(CLStatus& s)
	{
		if(battr)
			if(pthread_condattr_destroy(&attr) != 0)
				cout << "pthread_condattr_destroy error" << endl;

		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}