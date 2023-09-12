#include <iostream>
#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLMessageQueueBySTLqueue.h"
#include "CLMessage.h"
#include "CLMsgLoopManagerForSTLqueue.h"
#include "CLMessageObserver.h"
#include "CLExecutiveFunctionForMsgLoop.h"

using namespace std;

#define ADD_MSG 0
#define QUIT_MSG 1

class CLAddMsgProcessor;

class CLAddMessage : public CLMessage
{
public:
	friend class CLMyMsgProcessor;

	CLAddMessage(int Op1, int Op2):CLMessage(ADD_MSG)
	{
		m_Op1 = Op1;
		m_Op2 = Op2;
	}

	virtual ~CLAddMessage()
	{
	}

private:
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
		cout << "quit..." << endl;

		return CLStatus(QUIT_MESSAGE_LOOP, 0);
	}
};

int main()
{
	CLMessageQueueBySTLqueue *pQ = new CLMessageQueueBySTLqueue();
	CLThread *t = new CLThread(new CLExecutiveFunctionForMsgLoop(new CLMsgLoopManagerForSTLqueue(new CLMyMsgProcessor(), pQ)), true);
	t->Run(0);

	CLAddMessage *paddmsg = new CLAddMessage(2, 4);
	pQ->PushMessage(paddmsg);

	CLAddMessage *paddmsg1 = new CLAddMessage(3, 6);
	pQ->PushMessage(paddmsg1);

	CLAddMessage *paddmsg2 = new CLAddMessage(5, 6);
	pQ->PushMessage(paddmsg2);

	CLQuitMessage *pquitmsg = new CLQuitMessage();
	pQ->PushMessage(pquitmsg);

	t->WaitForDeath();

	return 0;
}
