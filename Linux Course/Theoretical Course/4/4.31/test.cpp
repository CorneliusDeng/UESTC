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

		CLEvent event("SharedEvent");

		CLExecutive *process = new CLProcess(new CLProcessFunctionForExec, true);
		if(!((process->Run((void *)"./test/a.out")).IsSuccess()))
		{
			cout << "Run error" << endl;
			throw CLStatus(-1, 0);
		}

		if(!event.Wait().IsSuccess())
			cout << "Wait error" << endl;

		cout << "in parent" << endl;

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