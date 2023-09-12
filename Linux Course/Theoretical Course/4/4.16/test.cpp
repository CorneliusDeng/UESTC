#include <iostream>
#include <unistd.h>
#include "LibExecutive.h"
#include "CLProcess.h"
#include "CLProcessFunctionForExec.h"

using namespace std;

int main()
{
	if(!CLLibExecutiveInitializer::Initialize().IsSuccess())
	{
		cout << "Initialize error" << endl;
		return 0;
	}

	CLExecutive *p = new CLProcess(new CLProcessFunctionForExec, true);
	if(p->Run((void *)"./test/a.out hello world").IsSuccess())
	{
		sleep(30);
		p->WaitForDeath();
	}

	if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
		cout << "Destroy error" << endl;

	return 0;
}
