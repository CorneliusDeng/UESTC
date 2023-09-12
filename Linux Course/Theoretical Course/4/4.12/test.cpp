#include <iostream>
#include "CLProcess.h"
#include "CLProcessFunctionForExec.h"

using namespace std;

int main()
{
	CLExecutive *p = new CLProcess(new CLProcessFunctionForExec, true);
	if(p->Run((void *)"./test/a.out hello world").IsSuccess())
	{
		p->WaitForDeath();
	}

	return 0;
}
