#ifndef CLTHREAD_H
#define CLTHREAD_H

#include <pthread.h>
#include "CLExecutive.h"
#include "CLStatus.h"

class CLThread : public CLExecutive
{
public:
	explicit CLThread(CLCoordinator *pCoordinator);
	virtual ~CLThread();

	virtual CLStatus Run();

	virtual CLStatus WaitForDeath();

private:
	static void* StartFunctionOfThread(void *pContext);

private:
	pthread_t m_ThreadID; 
};

#endif
