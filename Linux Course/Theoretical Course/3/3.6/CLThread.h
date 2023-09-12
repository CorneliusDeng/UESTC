#ifndef CLTHREAD_H
#define CLTHREAD_H

#include <pthread.h>
#include "CLStatus.h"

class CLThread
{
public:
	CLThread();
	~CLThread();

	CLStatus Run(void *pContext = 0);
	CLStatus WaitForDeath();

private:
	static void* StartFunctionOfThread(void *pContext);

private:
	CLStatus RunThreadFunction();

private:
	void *m_pContext;
	pthread_t m_ThreadID;
};

#endif
