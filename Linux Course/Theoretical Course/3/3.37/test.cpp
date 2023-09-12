#include <iostream>
#include "CLThread.h"
#include "CLMessage.h"
#include "CLMsgLoopManagerForSTLqueue.h"
#include "CLMessageObserver.h"
#include "CLExecutiveFunctionForMsgLoop.h"
#include "CLExecutiveNameServer.h"
#include "CLExecutiveCommunication.h"

using namespace std;

#define ADD_MSG 0
#define QUIT_MSG 1
#define DESTROY_MSG 2

class CLDestroyMessage : public CLMessage
{
public:
	CLDestroyMessage() : CLMessage(DESTROY_MSG)
	{
	}

	virtual ~CLDestroyMessage()
	{
	}
};

class CLAddMessage : public CLMessage
{
public:
	CLAddMessage(int Op1, int Op2) : CLMessage(ADD_MSG)
	{
		m_Op1 = Op1;
		m_Op2 = Op2;
	}

	virtual ~CLAddMessage()
	{
		if(!(CLExecutiveNameServer::PostExecutiveMessage("destroy", new CLDestroyMessage())).IsSuccess())
			cout << "~CLAddMessage error" << endl;
	}

	int m_Op1;
	int m_Op2;
};

class CLQuitMessage : public CLMessage
{
public:
	CLQuitMessage() : CLMessage(QUIT_MSG)
	{
	}

	virtual ~CLQuitMessage()
	{
	}
};

class CLMyMsgProcessor : public CLMessageObserver
{
public:
	CLMyMsgProcessor()
	{
	}

	virtual ~CLMyMsgProcessor()
	{
	}

	virtual CLStatus Initialize(CLMessageLoopManager *pMessageLoop, void* pContext)
	{
		pMessageLoop->Register(ADD_MSG, (CallBackForMessageLoop)(&CLMyMsgProcessor::On_AddMsg));
		pMessageLoop->Register(QUIT_MSG, (CallBackForMessageLoop)(&CLMyMsgProcessor::On_QuitMsg));

		return CLStatus(0, 0);
	}

	CLStatus On_AddMsg(CLMessage *pM)
	{
		CLAddMessage *pAddMsg = (CLAddMessage *)pM;

		cout << pAddMsg->m_Op1 + pAddMsg->m_Op2 << endl;

		return CLStatus(0, 0);
	}	

	CLStatus On_QuitMsg(CLMessage *pM)
	{
		cout << "adder quit..." << endl;

		return CLStatus(QUIT_MESSAGE_LOOP, 0);
	}
};

class CLDestroyMsgProcessor : public CLMessageObserver
{
public:
	CLDestroyMsgProcessor()
	{
	}

	virtual ~CLDestroyMsgProcessor()
	{
	}

	virtual CLStatus Initialize(CLMessageLoopManager *pMessageLoop, void* pContext)
	{
		pMessageLoop->Register(DESTROY_MSG, (CallBackForMessageLoop)(&CLDestroyMsgProcessor::On_DestroyMsg));
		pMessageLoop->Register(QUIT_MSG, (CallBackForMessageLoop)(&CLDestroyMsgProcessor::On_QuitMsg));

		return CLStatus(0, 0);
	}

	CLStatus On_DestroyMsg(CLMessage *pM)
	{
		cout << "in On_DestroyMsg" << endl;
		return CLStatus(0, 0);
	}	

	CLStatus On_QuitMsg(CLMessage *pM)
	{
		cout << "destroy quit..." << endl;

		return CLStatus(QUIT_MESSAGE_LOOP, 0);
	}
};

int main()
{
	CLThread *t = new CLThread(new CLExecutiveFunctionForMsgLoop(new CLMsgLoopManagerForSTLqueue(new CLMyMsgProcessor, "adder")), true);
	t->Run(0);

	CLThread *t2 = new CLThread(new CLExecutiveFunctionForMsgLoop(new CLMsgLoopManagerForSTLqueue(new CLDestroyMsgProcessor, "destroy")), true);
	t2->Run(0);

	sleep(2);

	CLExecutiveNameServer::PostExecutiveMessage("adder", new CLAddMessage(2, 4));
	CLExecutiveNameServer::PostExecutiveMessage("adder", new CLAddMessage(3, 6));
	CLExecutiveNameServer::PostExecutiveMessage("adder", new CLQuitMessage());

	CLExecutiveNameServer::PostExecutiveMessage("adder", new CLAddMessage(5, 6));

	t->WaitForDeath();

	CLExecutiveNameServer::PostExecutiveMessage("destroy", new CLQuitMessage());

	t2->WaitForDeath();

	return 0;
}
