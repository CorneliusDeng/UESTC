#include "CLThread.h"
#include "CLCoordinator.h"
#include "CLLogger.h"

CLThread::CLThread(CLCoordinator *pCoordinator) : CLExecutive(pCoordinator)
{
}

CLThread::~CLThread()
{
}

CLStatus CLThread::Run()
{	
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

	pThreadThis->m_pCoordinator->ReturnControlRights();

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
