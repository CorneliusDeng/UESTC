#include "CLMessageLoopManager.h"
#include "CLMessageObserver.h"
#include "CLMessage.h"
#include "CLLogger.h"

CLMessageLoopManager::CLMessageLoopManager()
{
}

CLMessageLoopManager::~CLMessageLoopManager()
{
}

CLStatus CLMessageLoopManager::Register(unsigned long lMsgID, CLMessageObserver *pMsgObserver)
{
	m_MsgMappingTable[lMsgID] = pMsgObserver;

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

	std::map<unsigned long, CLMessageObserver*>::iterator iter1;
	for(iter1 = m_MsgMappingTable.begin(); iter1 != m_MsgMappingTable.end(); iter1++)
	{
		iter1->second->Initialize(pContext);
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

	std::map<unsigned long, CLMessageObserver*>::iterator iter2;
	for(iter2 = m_MsgMappingTable.begin(); iter2 != m_MsgMappingTable.end(); iter2++)
	{
		delete iter2->second;
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
	std::map<unsigned long, CLMessageObserver*>::iterator it;
	it = m_MsgMappingTable.find(pMessage->m_clMsgID);

	if(it == m_MsgMappingTable.end())
	{
		CLLogger::WriteLogMsg("In CLMessageLoopManager::DispatchMessage(), it == m_MsgMappingTable.end", 0);
		return CLStatus(-1, 0);
	}

	if(it->second != 0)
		return it->second->Notify(pMessage);
	else
	{
		CLLogger::WriteLogMsg("In CLMessageLoopManager::DispatchMessage(), it->second == 0", 0);
		return CLStatus(-1, 0);
	}
}