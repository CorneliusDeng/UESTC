#ifndef CLMESSAGEOBSERVER_H
#define CLMESSAGEOBSERVER_H

#include "CLStatus.h"

class CLMessage;

class CLMessageObserver
{
public:
	CLMessageObserver();
	virtual ~CLMessageObserver();

	virtual CLStatus Initialize(void* pContext) = 0;
	virtual CLStatus DispatchMessage(CLMessage *pMsg) = 0;

private:
	CLMessageObserver(const CLMessageObserver&);
	CLMessageObserver& operator=(const CLMessageObserver&);
};

#endif
