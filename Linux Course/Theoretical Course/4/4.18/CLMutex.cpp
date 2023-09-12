#include "CLMutex.h"
#include "CLLogger.h"

CLMutex::CLMutex()
{
	m_pMutex = new pthread_mutex_t;
	
	m_bNeededDestroy = true;

	int  r = pthread_mutex_init(m_pMutex, 0);
	if(r != 0)
	{
		delete m_pMutex;

		CLLogger::WriteLogMsg("In CLMutex::CLMutex(), pthread_mutex_init error", r);
		throw "In CLMutex::CLMutex(), pthread_mutex_init error";
	}
}

CLMutex::CLMutex(pthread_mutex_t *pMutex)
{
	if(pMutex == 0)
		throw "In CLMutex::CLMutex(), pMutex is 0";

	m_pMutex = pMutex;
	m_bNeededDestroy = false;
}

CLMutex::~CLMutex()
{
	if(m_bNeededDestroy)
	{
		int r = pthread_mutex_destroy(m_pMutex);
		if(r != 0)
		{
			delete m_pMutex;

			CLLogger::WriteLogMsg("In CLMutex::~CLMutex(), pthread_mutex_destroy error", r);
			throw "In CLMutex::~CLMutex(), pthread_mutex_destroy error";
		}

		delete m_pMutex;
	}
}

CLStatus CLMutex::Lock()
{
	int r = pthread_mutex_lock(m_pMutex);
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
	int r = pthread_mutex_unlock(m_pMutex);
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