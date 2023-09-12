#ifndef CLMessageLoopManager_H
#define CLMessageLoopManager_H

#include "CLStatus.h"

class CLMessage;
class CLMessageObserver;

#define QUIT_MESSAGE_LOOP 1

class CLMessageLoopManager
{
public:
	CLMessageLoopManager(CLMessageObserver *pMessageObserver);
	virtual ~CLMessageLoopManager();

	virtual CLStatus EnterMessageLoop(void *pContext);

protected:
	virtual CLStatus Initialize() = 0;
	virtual CLStatus Uninitialize() = 0;
	
	virtual CLMessage* WaitForMessage() = 0;

private:
	CLMessageLoopManager(const CLMessageLoopManager&);
	CLMessageLoopManager& operator=(const CLMessageLoopManager&);

protected:
	CLMessageObserver *m_pMessageObserver;
};

#endif
