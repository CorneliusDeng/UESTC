#include "CLMessageLoopManager.h"
#include "CLMessage.h"
#include "CLLogger.h"

CLMessageLoopManager::CLMessageLoopManager()
{
}

CLMessageLoopManager::~CLMessageLoopManager()
{
}

CLStatus CLMessageLoopManager::EnterMessageLoop(void *pContext)
{
	CLStatus s = Initialize();
	if(!s.IsSuccess())
	{
		CLLogger::WriteLogMsg("In CLMessageLoopManager::EnterMessageLoop(), Initialize error", 0);
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
		
		CLStatus s3 = DispatchMessage(pMsg);

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
