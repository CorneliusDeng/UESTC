#ifndef MESSAGE_HEAD
#define MESSAGE_HEAD

#include "LibExecutive.h"

#define TEST_MESSAGE_ID 1
#define QUIT_MESSAGE_ID 2

class CLTestMsg : public CLMessage
{
public:
	CLTestMsg() : CLMessage(TEST_MESSAGE_ID)
	{
		i = 2;
		j = 3;
	}

	int i;
	int j;
};

class CLQuitMsg : public CLMessage
{
public:
	CLQuitMsg() : CLMessage(QUIT_MESSAGE_ID)
	{
	}
};

class CLTestMsgSerializer : public CLMessageSerializer
{
public:
	virtual char *Serialize(CLMessage *pMsg, unsigned int *pFullLength, unsigned int HeadLength)
	{
		CLTestMsg *p = dynamic_cast<CLTestMsg *>(pMsg);
		if(p == 0)
		{
			cout << "dynamic_cast error" << endl;
			return 0;
		}

		*pFullLength = HeadLength + 8 + 4 + 4;
		char *pBuf = new char[*pFullLength];

		long *pID = (long *)(pBuf + HeadLength);
		*pID = p->m_clMsgID;

		int *pi = (int *)(pBuf + HeadLength + 8);
		*pi = p->i;

		int *pj = (int *)(pBuf + HeadLength + 8 + 4);
		*pj = p->j;

		return pBuf;
	}
};

class CLQuitMsgSerializer : public CLMessageSerializer
{
public:
	virtual char *Serialize(CLMessage *pMsg, unsigned int *pFullLength, unsigned int HeadLength)
	{
		CLQuitMsg *p = dynamic_cast<CLQuitMsg *>(pMsg);
		if(p == 0)
		{
			cout << "dynamic_cast error" << endl;
			return 0;
		}

		*pFullLength = HeadLength + 8;
		char *pBuf = new char[*pFullLength];

		long *pID = (long *)(pBuf + HeadLength);
		*pID = p->m_clMsgID;

		return pBuf;
	}
};

class CLTestMsgDeserializer : public CLMessageDeserializer
{
public:
	virtual CLMessage *Deserialize(char *pBuffer)
	{
		long id = *((long *)pBuffer);
		if(id != TEST_MESSAGE_ID)
			return 0;

		CLTestMsg *p = new CLTestMsg;
		p->i = *((int *)(pBuffer + sizeof(id)));
		p->j = *((int *)(pBuffer + sizeof(id) + sizeof(int)));

		return p;
	}
};

class CLQuitMsgDeserializer : public CLMessageDeserializer
{
public:
	virtual CLMessage *Deserialize(char *pBuffer)
	{
		long id = *((long *)pBuffer);
		if(id != QUIT_MESSAGE_ID)
			return 0;

		CLQuitMsg *p = new CLQuitMsg;
		return p;
	}
};

#endif