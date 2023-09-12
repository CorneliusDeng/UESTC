#include <iostream>
#include <unistd.h>
#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLMutex.h"
#include "CLCriticalSection.h"
#include "CLConditionVariable.h"

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
	CLConditionVariable *pCondition = (CLConditionVariable *)pContext;
	//sleep(2);

	pCondition->Wakeup();

	return CLStatus(0, 0);
    }
};

int main()
{
    CLConditionVariable *pCondition = new CLConditionVariable();
    CLMutex *pMutex = new CLMutex();
    
    CLExecutiveFunctionProvider *myfunction = new CLMyFunction();
    CLExecutive *pThread = new CLThread(myfunction);
    pThread->Run((void *)pCondition);

    {
	CLCriticalSection cs(pMutex);

	pCondition->Wait(pMutex);
    }

    pThread->WaitForDeath();

    cout << "in main thread" << endl;

    delete pThread;
    delete myfunction;
    delete pMutex;
    delete pCondition;

    return 0;
}
