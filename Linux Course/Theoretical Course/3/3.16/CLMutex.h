#ifndef CLMutex_H
#define CLMutex_H

#include <pthread.h>
#include "CLStatus.h"

class CLMutex
{
public:
	/*
	构造函数和析构函数出错时，会抛出字符串类型异常
	*/
	CLMutex();
	virtual ~CLMutex();

	CLStatus Lock();
	CLStatus Unlock();

private:
	CLMutex(const CLMutex&);
	CLMutex& operator=(const CLMutex&);

private:
	pthread_mutex_t m_Mutex;
};

#endif