#include <iostream>
#include <unistd.h>
#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"

using namespace std;

class CLParaPrinter : public CLExecutiveFunctionProvider
{
public:
    CLParaPrinter()
    {
    }

    virtual ~CLParaPrinter()
    {
    }

    virtual CLStatus RunExecutiveFunction(void *pContext)
    {
	long i = (long)pContext;
	cout << i << endl;
	
	return CLStatus(0, 0);
    }
};

int main()
{
    CLExecutiveFunctionProvider *printer = new CLParaPrinter();
    CLExecutive *pThread = new CLThread(printer);

    pThread->Run((void *)2);
    pThread->WaitForDeath();

    delete pThread;
    delete printer;

    return 0;
}
