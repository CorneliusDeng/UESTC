#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLLogger.h"

CLThread::CLThread(CLExecutiveFunctionProvider *pExecutiveFunctionProvider) : CLExecutive(pExecutiveFunctionProvider)
{
	m_pContext = 0;
}

CLThread::~CLThread()
{
}

CLStatus CLThread::Run(void *pContext)
{	
	m_pContext = pContext;

	int r = pthread_create(&m_ThreadID, 0, StartFunctionOfThread, this);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLThread::Run(), pthread_create error", r);
		return CLStatus(-1, 0);
	}

	return CLStatus(0, 0);
}

void* CLThread::StartFunctionOfThread(void *pThis)
{
	CLThread *pThreadThis = (CLThread *)pThis;

	pThreadThis->m_pExecutiveFunctionProvider->RunExecutiveFunction(pThreadThis->m_pContext);

	return 0;
}

CLStatus CLThread::WaitForDeath()
{
	int r = pthread_join(m_ThreadID, 0);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLThread::WaitForDeath(), pthread_join error", r);
		return CLStatus(-1, 0);
	}

	return CLStatus(0, 0);
}
