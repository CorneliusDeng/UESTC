#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include "CLLogger.h"

#define LOG_FILE_NAME "CLLogger.txt"
#define MAX_SIZE 265

CLLogger* CLLogger::m_pLog = 0;

CLLogger::CLLogger()
{
	m_Fd = open(LOG_FILE_NAME, O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR); 
}

CLStatus CLLogger::WriteLog(const char *pstrMsg, long lErrorCode)
{
	if(pstrMsg == 0)
		return CLStatus(-1, 0);
	
	if(strlen(pstrMsg) == 0)
		return CLStatus(-1, 0);

	if(m_Fd == -1)
		return CLStatus(-1, 0);

	ssize_t r = write(m_Fd, pstrMsg, strlen(pstrMsg));
	if(r == -1)
		return CLStatus(-1, errno);

	char buf[MAX_SIZE];
	snprintf(buf, MAX_SIZE, "	Error code: %ld\r\n",  lErrorCode);

	r = write(m_Fd, buf, strlen(buf));
	if(r == -1)
		return CLStatus(-1, errno);

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
