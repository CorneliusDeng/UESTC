#include <iostream>
#include "MessageHead.h"

using namespace std;

class CLTestObserver : public CLMessageObserver
{
public:
	CLTestObserver()
	{
		m_Child = NULL;
	}

	virtual ~CLTestObserver()
	{
		if(m_Child != NULL)
			m_Child->WaitForDeath();
	}

	virtual CLStatus Initialize(CLMessageLoopManager *pMessageLoop, void* pContext)
	{
		pMessageLoop->Register(TEST_MESSAGE_ID, (CallBackForMessageLoop)(&CLTestObserver::On_Test));
		pMessageLoop->Register(QUIT_MESSAGE_ID, (CallBackForMessageLoop)(&CLTestObserver::On_Quit));

		m_Child = new CLProcess(new CLProcessFunctionForExec, true);
		if(!((m_Child->Run((void *)"./test/a.out")).IsSuccess()))
		{
			cout << "Run error" << endl;
			m_Child = NULL;
		}

		return CLStatus(0, 0);
	}

	CLStatus On_Test(CLMessage *pm)
	{
		CLTestMsg *p = dynamic_cast<CLTestMsg*>(pm);
		if(p == 0)
			return CLStatus(0, 0);
		
		cout << p->m_clMsgID << endl;
		cout << p->i << endl;
		cout << p->j << endl;

		return CLStatus(0, 0);
	}

	CLStatus On_Quit(CLMessage *pm)
	{
		CLQuitMsg *p = dynamic_cast<CLQuitMsg*>(pm);
		if(p == 0)
			return CLStatus(0, 0);
		
		return CLStatus(QUIT_MESSAGE_LOOP, 0);
	}

private:
	CLExecutive *m_Child;
};

int main()
{
	try
	{
		if(!CLLibExecutiveInitializer::Initialize().IsSuccess())
		{
			cout << "Initialize error" << endl;
			return 0;
		}

		CLThreadForMsgLoop thread(new CLTestObserver, "test_pipe", true, EXECUTIVE_BETWEEN_PROCESS_USE_PIPE_QUEUE);
		thread.RegisterDeserializer(TEST_MESSAGE_ID, new CLTestMsgDeserializer);
		thread.RegisterDeserializer(QUIT_MESSAGE_ID, new CLQuitMsgDeserializer);

		thread.Run(0);

		throw CLStatus(0, 0);
	}
	catch(CLStatus& s)
	{
		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}