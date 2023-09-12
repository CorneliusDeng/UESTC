#include <iostream>
#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"

using namespace std;

struct SPara
{
    int Flag;
    pthread_mutex_t mutex;
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

	pthread_mutex_lock(&(p->mutex));

	p->Flag++;
   cout <<"This is child thread, p->flag ="<< p->Flag << endl;

	pthread_mutex_unlock(&(p->mutex));

	return CLStatus(0, 0);
    }
};

int main()
{
    CLExecutiveFunctionProvider *myfunction = new CLMyFunction();
    CLExecutive *pThread = new CLThread(myfunction);

    SPara *p = new SPara;
    p->Flag = 3;
    pthread_mutex_init(&(p->mutex), 0);

    pThread->Run((void *)p);

    pthread_mutex_lock(&(p->mutex));

    p->Flag++;
    cout <<"This is parent thread, p->flag ="<< p->Flag << endl;

    pthread_mutex_unlock(&(p->mutex));

    pThread->WaitForDeath();

    pthread_mutex_destroy(&(p->mutex));
    delete p;

    delete pThread;
    delete myfunction;

    return 0;
}
