#include "CLMsgLoopManagerForSTLqueue.h"
#include "CLMessageQueueBySTLqueue.h"

CLMsgLoopManagerForSTLqueue::CLMsgLoopManagerForSTLqueue(CLMessageObserver *pMsgObserver, CLMessageQueueBySTLqueue *pMsgQueue) : CLMessageLoopManager(pMsgObserver)
{
	if(pMsgQueue == 0)
		throw "In CLMsgLoopManagerForSTLqueue::CLMsgLoopManagerForSTLqueue(), pMsgQueue error";

	m_pMsgQueue = pMsgQueue;
}

CLMsgLoopManagerForSTLqueue::~CLMsgLoopManagerForSTLqueue()
{
	delete m_pMsgQueue;
}

CLStatus CLMsgLoopManagerForSTLqueue::Initialize()
{
	return CLStatus(0, 0);
}

CLStatus CLMsgLoopManagerForSTLqueue::Uninitialize()
{
	return CLStatus(0, 0);
}
	
CLMessage* CLMsgLoopManagerForSTLqueue::WaitForMessage()
{
	return m_pMsgQueue->GetMessage();
}