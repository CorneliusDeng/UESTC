#ifndef CLTHREAD_H
#define CLTHREAD_H

#include <pthread.h>
#include "CLStatus.h"

class CLThread
{
public:
	CLThread();
	virtual ~CLThread();

	CLStatus Run(void *pContext = 0);
	CLStatus WaitForDeath();

private:
	static void* StartFunctionOfThread(void *pContext);

protected:
	virtual CLStatus RunThreadFunction() = 0;

	void *m_pContext;
	pthread_t m_ThreadID;
};

#endif
