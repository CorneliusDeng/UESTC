#include <iostream>
#include "LibExecutive.h"

using namespace std;

int main()
{
	CLSharedMemory sm("SharedMemoryForTest", 4);
	int *pAddr = (int *)sm.GetAddress();

	cout << "in child: " << *pAddr << endl;

	return 0;
}