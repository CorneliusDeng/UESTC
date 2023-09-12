#include "CLMessageLoopManager.h"
#include "CLMessageObserver.h"
#include "CLMessage.h"
#include "CLLogger.h"

CLMessageLoopManager::CLMessageLoopManager(CLMessageObserver *pMessageObserver)
{
	if(pMessageObserver == 0)
		throw "In CLMessageLoopManager::CLMessageLoopManager(), pMessageObserver error";

	m_pMessageObserver = pMessageObserver;
}

CLMessageLoopManager::~CLMessageLoopManager()
{
	delete m_pMessageObserver;
}

CLStatus CLMessageLoopManager::EnterMessageLoop(void *pContext)
{
	CLStatus s = Initialize();
	if(!s.IsSuccess())
	{
		CLLogger::WriteLogMsg("In CLMessageLoopManager::EnterMessageLoop(), Initialize error", 0);
		return CLStatus(-1, 0);
	}

	CLStatus s1 = m_pMessageObserver->Initialize(pContext);
	if(!s1.IsSuccess())
	{
		CLLogger::WriteLogMsg("In CLMessageLoopManager::EnterMessageLoop(), m_pMessageObserver->Initialize error", 0);

		CLStatus s2 = Uninitialize();
		if(!s2.IsSuccess())
			CLLogger::WriteLogMsg("In CLMessageLoopManager::EnterMessageLoop(), Uninitialize() error", 0);	
		
		return CLStatus(-1, 0);
	}
	
	while(true)
	{
		CLMessage *pMsg = WaitForMessage();
		if(pMsg == 0)
		{
			CLLogger::WriteLogMsg("In CLMessageLoopManager::EnterMessageLoop(), pMsg == 0", 0);
			continue;
		}
		
		CLStatus s3 = m_pMessageObserver->DispatchMessage(pMsg);

		delete pMsg;

		if(s3.m_clReturnCode == QUIT_MESSAGE_LOOP)
			break;
	}

	CLStatus s4 = Uninitialize();
	if(!s4.IsSuccess())
	{
		CLLogger::WriteLogMsg("In CLMessageLoopManager::EnterMessageLoop(), Uninitialize() error", 0);
		return CLStatus(-1, 0);
	}

	return CLStatus(0, 0);
}
