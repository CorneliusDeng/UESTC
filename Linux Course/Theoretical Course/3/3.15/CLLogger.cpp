#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "CLLogger.h"

#define LOG_FILE_NAME "CLLogger.txt"
#define MAX_SIZE 265
#define BUFFER_SIZE_LOG_FILE 4096

CLLogger* CLLogger::m_pLog = 0;
pthread_mutex_t *CLLogger::m_pMutexForCreatingLogger = CLLogger::InitializeMutex();

pthread_mutex_t *CLLogger::InitializeMutex()
{
	pthread_mutex_t *p = new pthread_mutex_t;

	if(pthread_mutex_init(p, 0) != 0)
	{
		delete p;
		return 0;
	}

	return p;
}

CLLogger::CLLogger()
{
	m_Fd = open(LOG_FILE_NAME, O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR); 
	if(m_Fd == -1)
		throw "In CLLogger::CLLogger(), open error";

	m_pLogBuffer = new char[BUFFER_SIZE_LOG_FILE];
	m_nUsedBytesForBuffer = 0;

	m_bFlagForProcessExit = false;

	m_pMutexForWritingLog = new pthread_mutex_t;
	if(pthread_mutex_init(m_pMutexForWritingLog, 0) != 0)
	{
		delete m_pMutexForWritingLog;
		delete [] m_pLogBuffer;
		close(m_Fd);

		throw "In CLLogger::CLLogger(), pthread_mutex_init error";
	}
}

CLStatus CLLogger::WriteLogMsg(const char *pstrMsg, long lErrorCode)
{
	CLLogger *pLog = CLLogger::GetInstance();
	if(pLog == 0)
		return CLStatus(-1, 0);
	
	CLStatus s = pLog->WriteLog(pstrMsg, lErrorCode);
	if(s.IsSuccess())
		return CLStatus(0, 0);
	else
		return CLStatus(-1, 0);
}

CLStatus CLLogger::Flush()
{
	if(pthread_mutex_lock(m_pMutexForWritingLog) != 0)
		return CLStatus(-1, 0);

	try
	{
	    if(m_nUsedBytesForBuffer == 0)
		    throw CLStatus(0, 0);

		if(write(m_Fd, m_pLogBuffer, m_nUsedBytesForBuffer) == -1)
			throw CLStatus(-1, errno);

		m_nUsedBytesForBuffer = 0;

		throw CLStatus(0, 0);
	}
	catch(CLStatus &s)
	{
		if(pthread_mutex_unlock(m_pMutexForWritingLog) != 0)
			return CLStatus(-1, 0);
		
	    return s;
	}
}

CLStatus CLLogger::WriteMsgAndErrcodeToFile(const char *pstrMsg, const char *pstrErrcode)
{
    if(write(m_Fd, pstrMsg, strlen(pstrMsg)) == -1)
		return CLStatus(-1, errno);

	if(write(m_Fd, pstrErrcode, strlen(pstrErrcode)) == -1)
		return CLStatus(-1, errno);

	return CLStatus(0, 0);
}

CLStatus CLLogger::WriteLog(const char *pstrMsg, long lErrorCode)
{
	if(pstrMsg == 0)
		return CLStatus(-1, 0);

	if(strlen(pstrMsg) == 0)
		return CLStatus(-1, 0);

	char buf[MAX_SIZE];
	snprintf(buf, MAX_SIZE, "	Error code: %ld\r\n",  lErrorCode);

	int len_strmsg = strlen(pstrMsg);
	int len_code = strlen(buf);
	unsigned int total_len = len_strmsg + len_code;

	if(pthread_mutex_lock(m_pMutexForWritingLog) != 0)
		return CLStatus(-1, 0);
	
	try
	{
		if((total_len > BUFFER_SIZE_LOG_FILE) || (m_bFlagForProcessExit))
			throw WriteMsgAndErrcodeToFile(pstrMsg, buf);

		unsigned int nleftroom = BUFFER_SIZE_LOG_FILE - m_nUsedBytesForBuffer;
		if(total_len > nleftroom)
		{
			if(write(m_Fd, m_pLogBuffer, m_nUsedBytesForBuffer) == -1)
				throw CLStatus(-1, errno);

			m_nUsedBytesForBuffer = 0;
		}

		memcpy(m_pLogBuffer + m_nUsedBytesForBuffer, pstrMsg, len_strmsg);
		m_nUsedBytesForBuffer += len_strmsg;

		memcpy(m_pLogBuffer + m_nUsedBytesForBuffer, buf, len_code);
		m_nUsedBytesForBuffer += len_code;

		throw CLStatus(0, 0);
	}
	catch (CLStatus& s)
	{
		if(pthread_mutex_unlock(m_pMutexForWritingLog) != 0)
			return CLStatus(-1, 0);

		return s;
	}
}

CLLogger* CLLogger::GetInstance()
{
	if(m_pLog != 0)
		return m_pLog;
	
	if(m_pMutexForCreatingLogger == 0)
		return 0;

	if(pthread_mutex_lock(m_pMutexForCreatingLogger) != 0)
		return 0;

	if(m_pLog == 0)
	{
		try
		{
			m_pLog = new CLLogger;
		}
		catch(const char*)
		{
			pthread_mutex_unlock(m_pMutexForCreatingLogger);
			return 0;
		}
		
		if(atexit(CLLogger::OnProcessExit) != 0)
		{
			m_pLog->m_bFlagForProcessExit = true;

			if(pthread_mutex_unlock(m_pMutexForCreatingLogger) != 0)
				return 0;

			m_pLog->WriteLog("In CLLogger::GetInstance(), atexit error", errno);
		}
		else
			if(pthread_mutex_unlock(m_pMutexForCreatingLogger) != 0)
				return 0;

		return m_pLog;
	}

	if(pthread_mutex_unlock(m_pMutexForCreatingLogger) != 0)
		return 0;

	return m_pLog;
}

void CLLogger::OnProcessExit()
{
	CLLogger *pLogger = CLLogger::GetInstance();
	if(pLogger != 0)
	{
		pthread_mutex_lock(pLogger->m_pMutexForWritingLog);
		pLogger->m_bFlagForProcessExit = true;
		pthread_mutex_unlock(pLogger->m_pMutexForWritingLog);

		pLogger->Flush();
	}
}