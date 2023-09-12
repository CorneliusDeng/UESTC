#include <iostream>
#include <unistd.h>
#include "CLThread.h"

using namespace std;

class CLMyThread : public CLThread
{
public:
    virtual CLStatus RunThreadFunction()
    {
	long i = (long)m_pContext;
	cout << i << endl;
	return CLStatus(0, 0);
    }
};

int main()
{
    CLThread *pThread = new CLMyThread;

    pThread->Run((void *)2);
    pThread->WaitForDeath();

    delete pThread;

    return 0;
}
