#include "CLPrivateExecutiveCommunicationByNamedPipe.h"

CLPrivateExecutiveCommunicationByNamedPipe::CLPrivateExecutiveCommunicationByNamedPipe(const char *pstrExecutiveName) : CLExecutiveCommunicationByNamedPipe(pstrExecutiveName, false)
{
}

CLPrivateExecutiveCommunicationByNamedPipe::~CLPrivateExecutiveCommunicationByNamedPipe()
{
}

char *CLPrivateExecutiveCommunicationByNamedPipe::GetMsgBuf(CLMessage *pMsg, unsigned int *pLength)
{
	int len = sizeof(CLMessage *);
	char *pBuf = new char[len];
	*((long *)pBuf) = (long)pMsg;

	*pLength = len;
	return pBuf;
}