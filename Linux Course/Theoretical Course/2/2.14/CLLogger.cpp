#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include "CLLogger.h"

#define LOG_FILE_NAME "CLLogger.txt"
#define MAX_SIZE 265
#define BUFFER_SIZE_LOG_FILE 4096

CLLogger* CLLogger::m_pLog = 0;

CLLogger::CLLogger()
{
	m_Fd = open(LOG_FILE_NAME, O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR); 

	m_pLogBuffer = new char[BUFFER_SIZE_LOG_FILE];
	m_nUsedBytesForBuffer = 0;
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
	if(m_Fd == -1)
		return CLStatus(-1, 0);

	if(m_pLogBuffer == 0)
		return CLStatus(-1, 0);

	if(m_nUsedBytesForBuffer == 0)
		return CLStatus(0, 0);
	
	ssize_t r = write(m_Fd, m_pLogBuffer, m_nUsedBytesForBuffer);
	if(r == -1)
		return CLStatus(-1, errno);

	m_nUsedBytesForBuffer = 0;

	return CLStatus(0, 0);
}

CLStatus CLLogger::WriteLog(const char *pstrMsg, long lErrorCode)
{
	if(pstrMsg == 0)
		return CLStatus(-1, 0);

	if(strlen(pstrMsg) == 0)
		return CLStatus(-1, 0);

	if(m_pLogBuffer == 0)
		return CLStatus(-1, 0);
	
	unsigned int nleftroom = BUFFER_SIZE_LOG_FILE - m_nUsedBytesForBuffer;

	unsigned int len_strmsg = strlen(pstrMsg);

	char buf[MAX_SIZE];
	snprintf(buf, MAX_SIZE, "	Error code: %ld\r\n",  lErrorCode);
	unsigned int len_code = strlen(buf);

	unsigned int total_len = len_code + len_strmsg;
	if(total_len > BUFFER_SIZE_LOG_FILE)
	{
		if(m_Fd == -1)
			return CLStatus(-1, 0);

		ssize_t r = write(m_Fd, pstrMsg, len_strmsg);
		if(r == -1)
			return CLStatus(-1, errno);

		r = write(m_Fd, buf, len_code);
		if(r == -1)
			return CLStatus(-1, errno);

		return CLStatus(0, 0);
	}

	if(total_len > nleftroom)
	{
		CLStatus s = Flush();
		if(!s.IsSuccess())
			return CLStatus(-1, 0);
	}

	memcpy(m_pLogBuffer + m_nUsedBytesForBuffer, pstrMsg, len_strmsg);

	m_nUsedBytesForBuffer += len_strmsg;

	memcpy(m_pLogBuffer + m_nUsedBytesForBuffer, buf, len_code);

	m_nUsedBytesForBuffer += len_code;
	
	return CLStatus(0, 0);
}

CLLogger* CLLogger::GetInstance()
{
	if(m_pLog == 0)
	{
		m_pLog = new CLLogger;
	}

	return m_pLog;
}
