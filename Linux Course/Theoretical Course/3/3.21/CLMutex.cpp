#include "CLMutex.h"
#include "CLLogger.h"

CLMutex::CLMutex()
{
	int  r = pthread_mutex_init(&m_Mutex, 0);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLMutex::CLMutex(), pthread_mutex_init error", r);
		throw "In CLMutex::CLMutex(), pthread_mutex_init error";
	}
}

CLMutex::~CLMutex()
{
	int r = pthread_mutex_destroy(&m_Mutex);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLMutex::~CLMutex(), pthread_mutex_destroy error", r);
		throw "In CLMutex::~CLMutex(), pthread_mutex_destroy error";
	}
}

CLStatus CLMutex::Lock()
{
	int r = pthread_mutex_lock(&m_Mutex);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLMutex::Lock(), pthread_mutex_lock error", r);
		return CLStatus(-1, 0);
	}
	else
	{
		return CLStatus(0, 0);
	}
}

CLStatus CLMutex::Unlock()
{
	int r = pthread_mutex_unlock(&m_Mutex);
	if(r != 0)
	{
		CLLogger::WriteLogMsg("In CLMutex::Unlock(), pthread_mutex_unlock error", r);
		return CLStatus(-1, 0);
	}
	else
	{
		return CLStatus(0, 0);
	}
}