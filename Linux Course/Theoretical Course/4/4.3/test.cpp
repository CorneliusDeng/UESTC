#include <iostream>
#include <unistd.h>
#include "CLMutex.h"

using namespace std;

CLMutex mutex;

int global = 0;

void test()
{
	mutex.Lock();

	global++;

	mutex.Unlock();
}

int main(void)
{
	mutex.Lock();

	if(fork() == 0)
	{
		test();
		return 0;
	}
	
	global++;

	mutex.Unlock();
	
	return 0;
}
