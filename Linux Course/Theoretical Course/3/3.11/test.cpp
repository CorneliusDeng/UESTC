#include <iostream>
#include "CLExecutiveFunctionProvider.h"
#include "CLThread.h"
#include "CLRegularCoordinator.h"

using namespace std;

class CLMyPrinter : public CLExecutiveFunctionProvider
{
public:
	CLMyPrinter(){}
	~CLMyPrinter(){}
	
	virtual CLStatus RunExecutiveFunction(void* pContext)
	{
		long i = (long)pContext;
		cout << i << endl;
		return CLStatus(0, 0);
	}	
};

int main()
{
	CLCoordinator *pCoordinator =  new CLRegularCoordinator();
	CLExecutive *pExecutive = new CLThread(pCoordinator);
	CLExecutiveFunctionProvider *pProvider = new CLMyPrinter();

	pCoordinator->SetExecObjects(pExecutive, pProvider);

	pCoordinator->Run((void *)5);

	pCoordinator->WaitForDeath();

	return 0;
}