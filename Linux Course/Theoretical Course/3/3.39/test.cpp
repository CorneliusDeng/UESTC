#include <iostream>
#include "CLThread.h"
#include "CLMessage.h"
#include "CLMsgLoopManagerForSTLqueue.h"
#include "CLMessageObserver.h"
#include "CLExecutiveFunctionForMsgLoop.h"
#include "CLExecutiveNameServer.h"
#include "CLExecutiveCommunication.h"
#include "CLThreadForMsgLoop.h"
#include "CLNonThreadForMsgLoop.h"

using namespace std;

class CLChildObserver : public CLMessageObserver
{
public:
	virtual CLStatus Initialize(CLMessageLoopManager *pMessageLoop, void* pContext)
	{
		pMessageLoop->Register(1, (CallBackForMessageLoop)(&CLChildObserver::On_1));

		CLExecutiveNameServer::PostExecutiveMessage("main", new CLMessage(1));

		return CLStatus(0, 0);
	}

	CLStatus On_1(CLMessage *pm)
	{
		cout << "in child On_1" << endl;
		return CLStatus(QUIT_MESSAGE_LOOP, 0);
	}
};

class CLMainObserver : public CLMessageObserver
{
private:
	CLThreadForMsgLoop *m_pTChild;

public:
	CLMainObserver()
	{
		m_pTChild = NULL;
	}

	virtual ~CLMainObserver()
	{
		delete m_pTChild;
	}

	virtual CLStatus Initialize(CLMessageLoopManager *pMessageLoop, void* pContext)
	{
		pMessageLoop->Register(1, (CallBackForMessageLoop)(&CLMainObserver::On_1));

		m_pTChild = new CLThreadForMsgLoop(new CLChildObserver, "child", true);

		m_pTChild->Run(0);

		return CLStatus(0, 0);
	}

	CLStatus On_1(CLMessage *pm)
	{
		cout << "in main On_1" << endl;
		CLExecutiveNameServer::PostExecutiveMessage("child", new CLMessage(1));
		return CLStatus(QUIT_MESSAGE_LOOP, 0);
	}
};

int main()
{
	CLNonThreadForMsgLoop p(new CLMainObserver, "main");
	p.Run(0);

	return 0;
}
