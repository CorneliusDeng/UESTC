#ifndef CLThreadForMsgLoop_H
#define CLThreadForMsgLoop_H

#include "CLStatus.h"

class CLMessageObserver;
class CLThread;

/************************************************************************/
/* CLThreadForMsgLoog类的分配释放问题，由使用者负责                     */
/* 该类将创建新线程，并使新线程进入消息循环中；当消息循环初始化完毕后才 */
/* Run方法才返回。														*/
/************************************************************************/
class CLThreadForMsgLoop
{
public:
	/*
	pMsgObserver应从堆中分配，且不必调用delete，pstrThreadName所代表的线程名称必须是唯一的
	默认情况下bWaitForDeath为false，若为true，则会在析构函数中等待新线程死亡
	*/
	CLThreadForMsgLoop(CLMessageObserver *pMsgObserver, const char *pstrThreadName);
	CLThreadForMsgLoop(CLMessageObserver *pMsgObserver, const char *pstrThreadName, bool bWaitForDeath);
	virtual ~CLThreadForMsgLoop();

	/*
	Run方法无论返回正确或出错，均只可调用一次。否则，另建一个CLThreadProxy对象
	*/
	CLStatus Run(void *pContext);

private:
	CLThreadForMsgLoop(const CLThreadForMsgLoop&);
	CLThreadForMsgLoop& operator=(const CLThreadForMsgLoop&);

private:
	CLThread *m_pThread;
	bool m_bWaitForDeath;
};

#endif