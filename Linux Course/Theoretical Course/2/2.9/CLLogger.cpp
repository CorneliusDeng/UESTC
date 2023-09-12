#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include "CLLogger.h"

#define LOG_FILE_NAME "CLLogger.txt"
#define MAX_SIZE 265

CLLogger::CLLogger()
{
}

CLLogger::~CLLogger()
{
}

CLStatus CLLogger::WriteLog(const char *pstrMsg, long lErrorCode)
{
	if(pstrMsg == 0)
		return CLStatus(-1, 0);

	if(strlen(pstrMsg) == 0)
		return CLStatus(-1, 0);
	
	int fd = open(LOG_FILE_NAME, O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR); 
	if(fd == -1)
		return CLStatus(-1, errno);
		
	ssize_t r = write(fd, pstrMsg, strlen(pstrMsg));
	if(r == -1)
	{
		close(fd);
		return CLStatus(-1, errno);
	}

	char buf[MAX_SIZE];
	snprintf(buf, MAX_SIZE, "	Error code: %ld\r\n",  lErrorCode);

	r = write(fd, buf, strlen(buf));
	if(r == -1)
	{
		close(fd);
		return CLStatus(-1, errno);
	}

	close(fd);

	return CLStatus(0, 0);
}
