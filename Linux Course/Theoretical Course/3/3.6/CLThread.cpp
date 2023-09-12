#include <iostream>
#include "CLThread.h"
#include "CLLogger.h"

using namespace std;

CLThread::CLThread()
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

void* CLThread::StartFunctionOfThread(void *pThis)
{
	CLThread *pThreadThis = (CLThread *)pThis;

	CLStatus s = pThreadThis->RunThreadFunction();

	return (void *)s.m_clReturnCode;
}

CLStatus CLThread::RunThreadFunction()
{
    cout << (long)m_pContext << endl;

	return CLStatus(0, 0);
}
