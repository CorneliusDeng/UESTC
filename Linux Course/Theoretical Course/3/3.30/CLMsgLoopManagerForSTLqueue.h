#ifndef CLMsgLoopManagerForSTLqueue_H
#define CLMsgLoopManagerForSTLqueue_H

#include "CLMessageLoopManager.h"

class CLMessageQueueBySTLqueue;

class CLMsgLoopManagerForSTLqueue : public CLMessageLoopManager
{
public:
	/*
	pMsgQueue应从堆中分配，且不必显示调用delete
	*/
	CLMsgLoopManagerForSTLqueue(CLMessageObserver *pMsgObserver, CLMessageQueueBySTLqueue *pMsgQueue);
	virtual ~CLMsgLoopManagerForSTLqueue();

protected:
	virtual CLStatus Initialize();
	virtual CLStatus Uninitialize();
	
	virtual CLMessage* WaitForMessage();

private:
	CLMsgLoopManagerForSTLqueue(const CLMsgLoopManagerForSTLqueue&);
	CLMsgLoopManagerForSTLqueue& operator=(const CLMsgLoopManagerForSTLqueue&);

private:
	CLMessageQueueBySTLqueue *m_pMsgQueue;
};

#endif