#include <iostream>
#include "CLThread.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLMessageQueueBySTLqueue.h"
#include "CLMessage.h"

using namespace std;

#define ADD_MSG 0
#define QUIT_MSG 1

class CLAdder;

class CLAddMessage : public CLMessage
{
public:
    	friend class CLAdder;

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

class CLAdder: public CLExecutiveFunctionProvider
{
public:
	CLAdder()
	{
	}

	virtual ~CLAdder()
	{
	}
	
	virtual CLStatus RunExecutiveFunction(void* pContext)
	{
		CLMessageQueueBySTLqueue *q = (CLMessageQueueBySTLqueue *)pContext;
		CLAddMessage *pAddMsg;

		while(1)
		{
			CLMessage *pM = q->GetMessage();
			switch(pM->m_clMsgID)
			{
			    case ADD_MSG:
				pAddMsg = (CLAddMessage *)pM;
				cout << pAddMsg->m_Op1 + pAddMsg->m_Op2 << endl;
				break;

			    case QUIT_MSG:
				cout << "quit..." << endl;
				return CLStatus(0, 0);

			    default:
				break;
			}
		}
	}
};

int main()
{
	CLMessageQueueBySTLqueue q;
	
	CLThread *t = new CLThread(new CLAdder, true);
	t->Run(&q);

	CLAddMessage addmsg(2, 4);
	q.PushMessage(&addmsg);

	CLAddMessage addmsg1(5, 4);
	q.PushMessage(&addmsg1);
	
	CLAddMessage addmsg2(2, 9);
	q.PushMessage(&addmsg2);

	CLQuitMessage quitmsg;
	q.PushMessage(&quitmsg);


	t->WaitForDeath();

	return 0;
}

