#ifndef CLNonThreadForMsgLoop_H
#define CLNonThreadForMsgLoop_H

#include "CLStatus.h"

class CLMessageObserver;
class CLExecutiveFunctionProvider;

/*
该类用于让线程直接进入消息循环，而不是创建新线程
*/
class CLNonThreadForMsgLoop
{
public:
	/*
	pMsgObserver应从堆中分配，且不必调用delete，pstrThreadName所代表的线程名称必须是唯一的
	*/
	CLNonThreadForMsgLoop(CLMessageObserver *pMsgObserver, const char *pstrThreadName);
	virtual ~CLNonThreadForMsgLoop();

	CLStatus Run(void *pContext);

private:
	CLNonThreadForMsgLoop(const CLNonThreadForMsgLoop&);
	CLNonThreadForMsgLoop& operator=(const CLNonThreadForMsgLoop&);

private:
	CLExecutiveFunctionProvider *m_pFunctionProvider;
};

#endif