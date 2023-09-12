#include <iostream>
#include <unistd.h>
#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLEvent.h"

using namespace std;

class CLMyFunction : public CLExecutiveFunctionProvider
{
public:
    CLMyFunction()
    {
    }

    virtual ~CLMyFunction()
    {
    }

    virtual CLStatus RunExecutiveFunction(void *pContext)
    {
	CLEvent *pEvent = (CLEvent *)pContext;

	//sleep(2);
	pEvent->Set();

	return CLStatus(0, 0);
    }
};

int main()
{
    CLEvent *pEvent = new CLEvent;
    
    CLExecutiveFunctionProvider *myfunction = new CLMyFunction();
    CLExecutive *pThread = new CLThread(myfunction);
    pThread->Run((void *)pEvent);

    pEvent->Wait();

    pThread->WaitForDeath();

    cout << "in main thread" << endl;

    delete pThread;
    delete myfunction;
    delete pEvent;

    return 0;
}
