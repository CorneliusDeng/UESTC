#include <iostream>
#include <unistd.h>
#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLMutex.h"
#include "CLCriticalSection.h"
#include "CLConditionVariable.h"

using namespace std;

struct SPara
{
    CLConditionVariable condition;
    CLMutex mutex;
    int flag;
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
		SPara *p = (SPara *)pContext;
		
		{
			CLCriticalSection cs(&(p->mutex));

			p->flag = 1;
		}

		p->condition.Wakeup();

		return CLStatus(0, 0);
    }
};

int main()
{
    SPara *p = new SPara;
    p->flag = 0;
    
    CLExecutiveFunctionProvider *myfunction = new CLMyFunction();
    CLExecutive *pThread = new CLThread(myfunction);
    pThread->Run((void *)p);

	{
		CLCriticalSection cs(&(p->mutex));

		while(p->flag == 0)
			p->condition.Wait(&(p->mutex));
	}

    pThread->WaitForDeath();

    cout << "in main thread" << endl;

    delete pThread;
    delete myfunction;
    delete p;

    return 0;
}
