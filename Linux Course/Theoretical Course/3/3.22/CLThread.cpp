#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLLogger.h"

CLThread::CLThread(CLExecutiveFunctionProvider *pExecutiveFunctionProvider) : CLExecutive(pExecutiveFunctionProvider)
{
	m_pContext = 0;
	m_bThreadCreated = false;
}

CLThread::~CLThread()
{
}

CLStatus CLThread::Run(void *pContext)
{
	if(m_bThreadCreated)
		return CLStatus(-1, 0);

	m_pContext = pContext;

	int r = pthread_create(&m_ThreadID, 0, StartFunctionOfThread, this);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLThread::Run(), pthread_create error", r);
		delete this;
		return CLStatus(-1, 0);
	}

	m_bThreadCreated = true;

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
	if(!m_bThreadCreated)
		return CLStatus(-1, 0);

	int r = pthread_join(m_ThreadID, 0);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLThread::WaitForDeath(), pthread_join error", r);
		return CLStatus(-1, 0);
	}

	delete this;

	return CLStatus(0, 0);
}