#include <iostream>
#include <unistd.h>
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

    void test()
    {
        throw 32;
    }

    virtual CLStatus RunExecutiveFunction(void *pContext)
    {
	try
	{
	    SPara *p = (SPara*)pContext;

	    p->mutex.Lock();

	    p->Flag++;

	    test();
	
	    p->mutex.Unlock();
	}
	catch(...)
	{
	    cout << "exception" << endl;
	}

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

    sleep(2);

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
