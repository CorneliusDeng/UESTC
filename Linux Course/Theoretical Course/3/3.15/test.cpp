#include <iostream>
#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLMutex.h"

using namespace std;

struct SPara
{
    int Flag;
    CLMutex mutex;
};

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
	SPara *p = (SPara*)pContext;

	p->mutex.Lock();

	p->Flag++;
	
	p->mutex.Unlock();

	return CLStatus(0, 0);
    }
};

int main()
{
    CLExecutiveFunctionProvider *myfunction = new CLMyFunction();
    CLExecutive *pThread = new CLThread(myfunction);

    SPara *p = new SPara;
    p->Flag = 3;

    pThread->Run((void *)p);

    p->mutex.Lock();

    p->Flag++;
    cout << p->Flag << endl;

    p->mutex.Unlock();

    pThread->WaitForDeath();

    delete p;

    delete pThread;
    delete myfunction;

    return 0;
}
