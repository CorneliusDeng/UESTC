#ifndef CLMessageLoopManager_H
#define CLMessageLoopManager_H

#include <map>
#include "CLStatus.h"

class CLMessage;
class CLMessageObserver;

#define QUIT_MESSAGE_LOOP 1

class CLMessageLoopManager
{
public:
	CLMessageLoopManager();
	virtual ~CLMessageLoopManager();

	virtual CLStatus EnterMessageLoop(void *pContext);
	virtual CLStatus Register(unsigned long lMsgID, CLMessageObserver *pMsgObserver);

protected:
	virtual CLStatus Initialize() = 0;
	virtual CLStatus Uninitialize() = 0;
	
	virtual CLMessage* WaitForMessage() = 0;
	virtual CLStatus DispatchMessage(CLMessage *pMessage);

private:
	CLMessageLoopManager(const CLMessageLoopManager&);
	CLMessageLoopManager& operator=(const CLMessageLoopManager&);

protected:
	std::map<unsigned long, CLMessageObserver*> m_MsgMappingTable;
};

#endif
