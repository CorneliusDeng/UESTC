#ifndef CLLogger_H
#define CLLogger_H

#include "CLStatus.h"

/*
用于向文件LOG_FILE_NAME中，记录日志信息
*/
class CLLogger
{
public:
	CLLogger();
	virtual ~CLLogger();

	CLStatus WriteLog(const char *pstrMsg, long lErrorCode);

private:
	CLLogger(const CLLogger&);
	CLLogger& operator=(const CLLogger&);

private:
	int m_Fd;
};

#endif