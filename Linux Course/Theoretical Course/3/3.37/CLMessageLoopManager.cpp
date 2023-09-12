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

CLStatus CLMessageLoopManager::Register(unsigned long lMsgID, CallBackForMessageLoop pMsgProcessFunction)
{
	if(pMsgProcessFunction == 0)
		return CLStatus(-1, 0);

	m_MsgMappingTable[lMsgID] = pMsgProcessFunction;

	return CLStatus(0, 0);
}

CLStatus CLMessageLoopManager::EnterMessageLoop(void *pContext)
{
	CLStatus s = Initialize();
	if(!s.IsSuccess())
	{
		CLLogger::WriteLogMsg("In CLMessageLoopManager::EnterMessageLoop(), Initialize error", 0);
		return CLStatus(-1, 0);
	}

	CLStatus s1 = m_pMessageObserver->Initialize(this, pContext);
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

CLStatus CLMessageLoopManager::DispatchMessage(CLMessage *pMessage)
{
	std::map<unsigned long, CallBackForMessageLoop>::iterator it;
	it = m_MsgMappingTable.find(pMessage->m_clMsgID);

	if(it == m_MsgMappingTable.end())
	{
		CLLogger::WriteLogMsg("In CLMessageLoopManager::DispatchMessage(), it == m_MsgMappingTable.end", 0);
		return CLStatus(-1, 0);
	}

	CallBackForMessageLoop pFunction = it->second;

	return (m_pMessageObserver->*pFunction)(pMessage);
}