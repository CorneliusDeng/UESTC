#ifndef CLMutex_H
#define CLMutex_H

#include <pthread.h>
#include "CLStatus.h"

class CLMutex
{
public:
	friend class CLConditionVariable;
	
	/*
	构造函数和析构函数出错时，会抛出字符串类型异常
	*/
	CLMutex();
	//需要保证pMutex指向的互斥量已经被初始化了
	explicit CLMutex(pthread_mutex_t *pMutex);
	virtual ~CLMutex();

	CLStatus Lock();
	CLStatus Unlock();

private:
	CLMutex(const CLMutex&);
	CLMutex& operator=(const CLMutex&);

private:
	pthread_mutex_t *m_pMutex;
	bool m_bNeededDestroy;
};

#endif