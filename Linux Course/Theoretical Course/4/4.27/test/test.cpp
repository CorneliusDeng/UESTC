#include <iostream>
#include "LibExecutive.h"

using namespace std;

int main()
{
	CLSharedMemory sm("SharedMemoryForTest");
	pthread_mutex_t *pmutex = (pthread_mutex_t *)sm.GetAddress();

	pthread_mutex_lock(pmutex);

	cout << "in child" << endl;

	pthread_mutex_unlock(pmutex);

	return 0;
}