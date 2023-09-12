#include<pthread.h>
#include<iostream>
#include<unistd.h>
#include<stdlib.h>
#include"CLStatus.h"
#include"CLLogger.h"

using namespace std;

class CMyFunction
{
public:
    CLStatus RunExecutiveFunction(void *pContext)
    {
	long i = (long)pContext;
	cout << i << endl;
	return CLStatus(0, 0);
    }
};


class CLThread
{
public:
    template<typename TDerivedClass>
    CLStatus Create()
    {
	pthread_t tid;
	int r = pthread_create(&tid, NULL, StartFunctionOfThread<TDerivedClass>, this);
	if(r != 0)
	{
	    CLLogger::WriteLogMsg("pthread_create error", r);
	    return CLStatus(-1, 0);
	}

	return CLStatus(0, 0);
    }

    template<typename TDerivedClass>
    static void* StartFunctionOfThread(void *pContext)
    {
	TDerivedClass *pThis = static_cast<TDerivedClass *>(pContext);
	
	pThis->ProcessByProxy();

	return 0;
    }
};

class CLProcess
{
public:
	template<typename TDerivedClass>
	CLStatus Create()
	{
		pid_t pid = fork();
		if(pid == 0)
		{
			TDerivedClass *pDerived = static_cast<TDerivedClass *>(this);
			pDerived->ProcessByProxy();
			exit(0);
		}

		if(pid == -1)
			return CLStatus(-1, 0);
		else
			return CLStatus(0, 0);
	}
};

template<typename TExecutive, typename TExecutiveFunctionProvider>
class CLCoordinator : public TExecutive, public TExecutiveFunctionProvider
{
public:
    CLStatus Run(void *pContext)
    {
	m_pContext = pContext;

	TExecutive *pExecutive = static_cast<TExecutive *>(this);

	typedef CLCoordinator<TExecutive, TExecutiveFunctionProvider> T;
	
	return (*pExecutive).template Create<T>();
    }

    void ProcessByProxy()
    {
	TExecutiveFunctionProvider *pProvider = static_cast<TExecutiveFunctionProvider*>(this);
	
	pProvider->RunExecutiveFunction(m_pContext);
    }

private:
    void *m_pContext;
};

int main()
{
    CLCoordinator<CLProcess, CMyFunction> t1;
    t1.Run((void *)5);
   
    return 0;
}
