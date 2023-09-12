#include "CLEvent.h"
#include "CLCriticalSection.h"
#include "CLLogger.h"

CLEvent::CLEvent()
{
	m_Flag = 0;
	m_bSemaphore = false;
}

CLEvent::CLEvent(bool bSemaphore)
{
	m_Flag = 0;
	m_bSemaphore = bSemaphore;
}

CLEvent::~CLEvent()
{
}

CLStatus CLEvent::Set()
{
	try
	{
		CLCriticalSection cs(&m_Mutex);

		m_Flag++;
	}
	catch(const char *str)
	{
		CLLogger::WriteLogMsg("In CLEvent::Set(), exception arise", 0);
		return CLStatus(-1, 0);
	}

	CLStatus s = m_Cond.Wakeup();
	if(!s.IsSuccess())
	{
		CLLogger::WriteLogMsg("In CLEvent::Set(), m_Cond.Wakeup error", 0);
		return CLStatus(-1, 0);
	}

	return CLStatus(0, 0);
}

CLStatus CLEvent::Wait()
{
	try
	{
		CLCriticalSection cs(&m_Mutex);

		while(m_Flag == 0)
		{
			CLStatus s = m_Cond.Wait(&m_Mutex);
			if(!s.IsSuccess())
			{
				CLLogger::WriteLogMsg("In CLEvent::Wait(), m_Cond.Wait error", 0);
				return CLStatus(-1, 0);
			}
		}

		if(m_bSemaphore)
		{
			m_Flag--;
		}
		else
		{
			m_Flag = 0;
		}
	}
	catch(const char* str)
	{
		CLLogger::WriteLogMsg("In CLEvent::Wait(), exception arise", 0);
		return CLStatus(-1, 0);
	}
	
	return CLStatus(0, 0);
}
