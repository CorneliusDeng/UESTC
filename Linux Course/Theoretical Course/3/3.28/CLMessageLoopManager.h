#ifndef CLMessageLoopManager_H
#define CLMessageLoopManager_H

#include "CLStatus.h"

class CLMessage;

#define QUIT_MESSAGE_LOOP 1

class CLMessageLoopManager
{
public:
	CLMessageLoopManager();
	virtual ~CLMessageLoopManager();

	virtual CLStatus EnterMessageLoop(void *pContext);

protected:
	/*
	初始化与反初始化消息循环，需要保证消息队列已经建立完毕
	*/
	virtual CLStatus Initialize() = 0;
	virtual CLStatus Uninitialize() = 0;
	
	virtual CLMessage* WaitForMessage() = 0;
	virtual CLStatus DispatchMessage(CLMessage *pMessage) = 0;

private:
	CLMessageLoopManager(const CLMessageLoopManager&);
	CLMessageLoopManager& operator=(const CLMessageLoopManager&);
};

#endif
