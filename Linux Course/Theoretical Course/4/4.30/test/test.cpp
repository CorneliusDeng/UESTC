#include <iostream>
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

		CLSharedMemory smFlag("SharedFlag");
		long *pFlag = (long *)smFlag.GetAddress();

		{
			CLMutex mutex("SharedMutex", MUTEX_USE_SHARED_PTHREAD);
			CLCriticalSection cs(&mutex);

			*pFlag = 1;
		}

		cout << "in child" << endl;
		sleep(5);

		CLConditionVariable cv("SharedCond");

		if(!(cv.Wakeup().IsSuccess()))
			cout << "Wakeup error" << endl;

		throw CLStatus(0, 0);
	}
	catch(CLStatus& s)
	{
		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}