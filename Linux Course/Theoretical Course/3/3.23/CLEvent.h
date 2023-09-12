#ifndef CLEVENT_H
#define CLEVENT_H

#include "CLStatus.h"
#include "CLMutex.h"
#include "CLConditionVariable.h"

/*
 创建一个初始无信号，自动重置信号的信号（用于唤醒一个 等待线程）
*/
class CLEvent
{
public: 
	/*
	构造函数和析构函数出错时，会抛出字符串类型异常
	*/
	CLEvent( );
	virtual ~CLEvent();

public: 
	CLStatus Set();

	CLStatus Wait();

private:
	CLEvent(const CLEvent&);
	CLEvent& operator=(const CLEvent&);

private:
	CLMutex m_Mutex;
	CLConditionVariable m_Cond;
	int m_Flag;
};

#endif