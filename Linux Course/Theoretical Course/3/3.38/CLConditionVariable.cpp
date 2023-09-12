#include "CLConditionVariable.h"
#include "CLMutex.h"
#include "CLLogger.h"

CLConditionVariable::CLConditionVariable()
{
	int  r = pthread_cond_init(&m_ConditionVariable, 0);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLConditionVariable::CLConditionVariable(), pthread_cond_init error", r);
		throw "In CLConditionVariable::CLConditionVariable(), pthread_cond_init error";
	}
}

CLConditionVariable::~CLConditionVariable()
{
	int r = pthread_cond_destroy(&m_ConditionVariable);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLConditionVariable::~CLConditionVariable(), pthread_cond_destroy error", r);
		throw "In CLConditionVariable::~CLConditionVariable(), pthread_cond_destroy error";
	}
}

CLStatus CLConditionVariable::Wait(CLMutex *pMutex)
{
	if(pMutex == NULL)
		return CLStatus(-1, 0);

	int r = pthread_cond_wait(&m_ConditionVariable, &(pMutex->m_Mutex));
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLConditionVariable::Wait, pthread_cond_wait error", r);
		return CLStatus(-1, 0);
	}
	else
	{
		return CLStatus(0, 0);
	}
}

CLStatus CLConditionVariable::Wakeup()
{
	int r = pthread_cond_signal(&m_ConditionVariable);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLConditionVariable::Wakeup, pthread_cond_signal error", r);
		return CLStatus(-1, 0);
	}
	else
	{
		return CLStatus(0, 0);
	}
}

CLStatus CLConditionVariable::WakeupAll()
{
	int r = pthread_cond_broadcast(&m_ConditionVariable);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLConditionVariable::WakeupAll, pthread_cond_broadcast error", r);
		return CLStatus(-1, 0);
	}
	else
	{
		return CLStatus(0, 0);
	}
}
