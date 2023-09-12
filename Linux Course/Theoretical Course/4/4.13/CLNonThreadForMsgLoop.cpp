#include <string.h>
#include "CLNonThreadForMsgLoop.h"
#include "CLExecutiveFunctionForMsgLoop.h"
#include "CLMsgLoopManagerForSTLqueue.h"
#include "CLLogger.h"
#include "CLThreadInitialFinishedNotifier.h"

CLNonThreadForMsgLoop::CLNonThreadForMsgLoop(CLMessageObserver *pMsgObserver, const char *pstrThreadName)
{
	if(pMsgObserver == 0)
		throw "In CLNonThreadForMsgLoop::CLNonThreadForMsgLoop(), pMsgObserver error";

	if((pstrThreadName == 0) || (strlen(pstrThreadName) == 0))
		throw "In CLNonThreadForMsgLoop::CLNonThreadForMsgLoop(), pstrThreadName error";

	m_pFunctionProvider = new CLExecutiveFunctionForMsgLoop(new CLMsgLoopManagerForSTLqueue(pMsgObserver, pstrThreadName));
}

CLNonThreadForMsgLoop::~CLNonThreadForMsgLoop()
{
	delete m_pFunctionProvider;
}

CLStatus CLNonThreadForMsgLoop::Run(void *pContext)
{
	CLThreadInitialFinishedNotifier notifier(0);		

	SLExecutiveInitialParameter para;
	para.pContext = pContext;
	para.pNotifier = &notifier;

	return m_pFunctionProvider->RunExecutiveFunction(&para);
}
