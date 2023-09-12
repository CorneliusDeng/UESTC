#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include "LibExecutive.h"
#include "CLProcess.h"
#include "CLProcessFunctionForExec.h"

using namespace std;

int main()
{
	bool battr = false;
	pthread_mutexattr_t attr;

	try
	{
		if(!CLLibExecutiveInitializer::Initialize().IsSuccess())
		{
			cout << "Initialize error" << endl;
			return 0;
		}
		
		if(pthread_mutexattr_init(&attr) != 0)
		{
			cout << "pthread_mutexattr_init error" << endl;
			throw CLStatus(-1, 0);
		}

		battr = true;

		if(pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED) != 0)
		{
			cout << "pthread_mutexattr_setpshared error" << endl;
			throw CLStatus(-1, 0);
		}

		CLSharedMemory sm("SharedMemoryForTest", sizeof(pthread_mutex_t));
		pthread_mutex_t *pmutex = (pthread_mutex_t *)sm.GetAddress();

		if(pthread_mutex_init(pmutex, &attr) != 0)
		{
			cout << "pthread_mutex_init error" << endl;
			throw CLStatus(-1, 0);
		}

		pthread_mutex_lock(pmutex);
		
		CLExecutive *process = new CLProcess(new CLProcessFunctionForExec, true);
		if(!(process->Run((void *)"./test/a.out")).IsSuccess())
			cout << "process Run error" << endl;
		else
		{
			cout << "in parent" << endl;
			sleep(5);

			pthread_mutex_unlock(pmutex);

			process->WaitForDeath();
		}

		pthread_mutex_destroy(pmutex);

		throw CLStatus(0, 0);
	}
	catch(CLStatus& s)
	{
		if(battr)
		{
			if(pthread_mutexattr_destroy(&attr) != 0)
				cout << "pthread_mutexattr_destroy error" << endl;
		}

		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}