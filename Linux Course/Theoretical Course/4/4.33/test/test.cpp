#include <iostream>
#include "../MessageHead.h"

using namespace std;

int main()
{
	try
	{
		if(!CLLibExecutiveInitializer::Initialize().IsSuccess())
		{
			cout << "Initialize error" << endl;
			return 0;
		}

		CLSharedExecutiveCommunicationByNamedPipe *pSender = new CLSharedExecutiveCommunicationByNamedPipe("test_pipe");
		pSender->RegisterSerializer(TEST_MESSAGE_ID, new CLTestMsgSerializer);
		pSender->RegisterSerializer(QUIT_MESSAGE_ID, new CLQuitMsgSerializer);

		CLExecutiveNameServer::GetInstance()->Register("test_pipe", pSender);

		CLTestMsg *pTestMsg = new CLTestMsg;
		pTestMsg->i = 3;
		pTestMsg->j = 4;
		CLExecutiveNameServer::PostExecutiveMessage("test_pipe", pTestMsg);

		CLQuitMsg *pQuitMsg = new CLQuitMsg;
		CLExecutiveNameServer::PostExecutiveMessage("test_pipe", pQuitMsg);

		CLExecutiveNameServer::GetInstance()->ReleaseCommunicationPtr("test_pipe");
		
		throw CLStatus(0, 0);
	}
	catch(CLStatus& s)
	{
		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}