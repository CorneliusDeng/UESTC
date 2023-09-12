#include <string.h>
#include "CLExecutiveNameServer.h"
#include "CLCriticalSection.h"
#include "CLLogger.h"
#include "CLExecutiveCommunication.h"
#include "CLMessage.h"

CLExecutiveNameServer* CLExecutiveNameServer::m_pNameServer = 0;
pthread_mutex_t *CLExecutiveNameServer::m_pMutex = CLExecutiveNameServer::InitializeMutex();

CLExecutiveNameServer::CLExecutiveNameServer()
{
}

CLExecutiveNameServer::~CLExecutiveNameServer()
{
}

CLStatus CLExecutiveNameServer::PostExecutiveMessage(const char* pstrExecutiveName, CLMessage *pMessage)
{
	if(pMessage == 0)
		return CLStatus(-1, 0);

	if((pstrExecutiveName == 0) || (strlen(pstrExecutiveName) == 0))
	{
		delete pMessage;
		return CLStatus(-1, 0);
	}

	CLExecutiveNameServer *pNameServer = CLExecutiveNameServer::GetInstance();
	if(pNameServer == 0)
	{
		CLLogger::WriteLogMsg("In CLExecutiveNameServer::PostExecutiveMessage(), CLExecutiveNameServer::GetInstance error", 0);
		delete pMessage;
		return CLStatus(-1, 0);
	}

	CLExecutiveCommunication* pComm = pNameServer->GetCommunicationPtr(pstrExecutiveName);
	if(pComm == 0)
	{
		CLLogger::WriteLogMsg("In CLExecutiveNameServer::PostExecutiveMessage(), pNameServer->GetCommunicationPtr error", 0);
		delete pMessage;
		return CLStatus(-1, 0);
	}	

	CLStatus s = pComm->PostExecutiveMessage(pMessage);
	if(!s.IsSuccess())
	{
		CLLogger::WriteLogMsg("In CLExecutiveNameServer::PostExecutiveMessage(), pComm->PostExecutiveMessage error", 0);

		CLStatus s1 = pNameServer->ReleaseCommunicationPtr(pstrExecutiveName);
		if(!s1.IsSuccess())
			CLLogger::WriteLogMsg("In CLExecutiveNameServer::PostExecutiveMessage(), pNameServer->ReleaseCommunicationPtr error", 0);

		return CLStatus(-1, 0);
	}

	CLStatus s2 = pNameServer->ReleaseCommunicationPtr(pstrExecutiveName);
	if(!s2.IsSuccess())
		CLLogger::WriteLogMsg("In CLExecutiveNameServer::PostExecutiveMessage(), pNameServer->ReleaseCommunicationPtr error", 0);

	return CLStatus(0, 0);
}

CLStatus CLExecutiveNameServer::Register(const char* strExecutiveName, CLExecutiveCommunication *pExecutiveCommunication)
{
	if(pExecutiveCommunication == 0)
		return CLStatus(-1, 0);

	if((strExecutiveName == 0) || (strlen(strExecutiveName) == 0))
	{
		delete pExecutiveCommunication;
		return CLStatus(-1, 0);
	}

	CLCriticalSection cs(&m_MutexForNameTable);
	
	std::map<std::string, SLExecutiveCommunicationPtrCount*>::iterator it = m_NameTable.find(strExecutiveName);	
	if(it != m_NameTable.end())
	{
		delete pExecutiveCommunication;
		CLLogger::WriteLogMsg("In CLExecutiveNameServer::Register(), m_NameTable.find error", 0);
		return CLStatus(-1, 0);
	}
	
	SLExecutiveCommunicationPtrCount *p = new SLExecutiveCommunicationPtrCount;
	p->pExecutiveCommunication = pExecutiveCommunication;
	p->nCount = 1;
	
	m_NameTable[strExecutiveName] = p;
	
	return CLStatus(0, 0);
}

CLExecutiveCommunication* CLExecutiveNameServer::GetCommunicationPtr(const char* strExecutiveName)
{
	if((strExecutiveName == 0) || (strlen(strExecutiveName) == 0))
		return 0;

	CLCriticalSection cs(&m_MutexForNameTable);

	std::map<std::string, SLExecutiveCommunicationPtrCount*>::iterator it = m_NameTable.find(strExecutiveName);
	if(it == m_NameTable.end())
	{
		CLLogger::WriteLogMsg("In CLExecutiveNameServer::GetCommunicationPtr(), m_NameTable.find error", 0);
		return 0;
	}

	it->second->nCount++;

	return it->second->pExecutiveCommunication;
}

CLStatus CLExecutiveNameServer::ReleaseCommunicationPtr(const char* strExecutiveName)
{
	if((strExecutiveName == 0) || (strlen(strExecutiveName) == 0))
		return CLStatus(-1, 0);

	CLCriticalSection cs(&m_MutexForNameTable);

	std::map<std::string, SLExecutiveCommunicationPtrCount*>::iterator it = m_NameTable.find(strExecutiveName);
	if(it == m_NameTable.end())
	{
		CLLogger::WriteLogMsg("In CLExecutiveNameServer::ReleaseCommunicationPtr(), m_NameTable.find error", 0);
		return CLStatus(-1, 0);
	}

	it->second->nCount--;

	if(it->second->nCount == 0)
	{
		delete it->second->pExecutiveCommunication;
		delete it->second;
		m_NameTable.erase(it);
	}
		
	return CLStatus(0, 0);
}

pthread_mutex_t *CLExecutiveNameServer::InitializeMutex()
{
	pthread_mutex_t *p = new pthread_mutex_t;

	int r = pthread_mutex_init(p, 0);
	if(r != 0)
	{
		delete p;
		return 0;
	}
	
	return p;
}

CLExecutiveNameServer* CLExecutiveNameServer::GetInstance()
{
	if(m_pNameServer == 0)
	{
		if(m_pMutex == 0)
			return 0;
		
		int r = pthread_mutex_lock(m_pMutex);
		if(r != 0)
			return 0;
		
		if(m_pNameServer == 0)
		{
			m_pNameServer = new CLExecutiveNameServer;
		}

		r = pthread_mutex_unlock(m_pMutex);
		if(r != 0)
			return 0;
	}

	return m_pNameServer;
}
