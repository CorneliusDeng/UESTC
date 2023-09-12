#include <iostream>
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

		CLSharedMemory sm("SharedMemoryForTest", 4);
		int *pAddr = (int *)sm.GetAddress();
		*pAddr = 5;
		
		CLExecutive *process = new CLProcess(new CLProcessFunctionForExec, true);
		if(!(process->Run((void *)"./test/a.out")).IsSuccess())
			cout << "process Run error" << endl;
		else
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