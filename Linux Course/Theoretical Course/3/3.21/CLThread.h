#ifndef CLTHREAD_H
#define CLTHREAD_H

#include <pthread.h>
#include "CLExecutive.h"
#include "CLStatus.h"

class CLThread : public CLExecutive
{
public:
	explicit CLThread(CLExecutiveFunctionProvider *pExecutiveFunctionProvider);
	virtual ~CLThread();

	virtual CLStatus Run(void *pContext = 0);

	virtual CLStatus WaitForDeath();

private:
	static void* StartFunctionOfThread(void *pContext);

private:
	void *m_pContext;
	pthread_t m_ThreadID; 
	bool m_bThreadCreated;
};

#endif
