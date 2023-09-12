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

	CLExecutive *p = new CLProcess(new CLProcessFunctionForExec, false);
	if(!p->Run((void *)"./test/a.out hello world").IsSuccess())
		cout << "Run error" << endl;

	if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
		cout << "Destroy error" << endl;

	return 0;
}
