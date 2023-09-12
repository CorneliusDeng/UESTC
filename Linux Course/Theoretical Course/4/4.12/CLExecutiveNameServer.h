#ifndef CLExecutiveNameServer_H
#define CLExecutiveNameServer_H

#include <pthread.h>
#include <map>
#include <string>
#include "CLStatus.h"
#include "CLMutex.h"

class CLExecutiveCommunication;
class CLMessage;

struct SLExecutiveCommunicationPtrCount
{
	CLExecutiveCommunication *pExecutiveCommunication;
	unsigned int nCount;
};

class CLExecutiveNameServer
{
public:
	/*
	出错时，构造函数和析构函数可能会产生字符串类型异常
	*/
	CLExecutiveNameServer();
	virtual ~CLExecutiveNameServer();

	static CLExecutiveNameServer* GetInstance();
	static CLStatus PostExecutiveMessage(const char* pstrExecutiveName, CLMessage *pMessage);

public:
	CLStatus Register(const char* strExecutiveName, CLExecutiveCommunication *pExecutiveCommunication);

	CLExecutiveCommunication* GetCommunicationPtr(const char* strExecutiveName);
	CLStatus ReleaseCommunicationPtr(const char* strExecutiveName);

private:
	CLExecutiveNameServer(const CLExecutiveNameServer&);
	CLExecutiveNameServer& operator=(const CLExecutiveNameServer&);

	static pthread_mutex_t *InitializeMutex();

private:
	static CLExecutiveNameServer *m_pNameServer;
	static pthread_mutex_t *m_pMutex;

private:
	std::map<std::string, SLExecutiveCommunicationPtrCount*> m_NameTable;
	CLMutex m_MutexForNameTable;
};

#endif
