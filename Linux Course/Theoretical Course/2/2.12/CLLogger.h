#ifndef CLLogger_H
#define CLLogger_H

#include "CLStatus.h"

/*
用于向文件LOG_FILE_NAME中，记录日志信息
*/
class CLLogger
{
public:
	static CLLogger* GetInstance();
	CLStatus WriteLog(const char *pstrMsg, long lErrorCode);

private:
	CLLogger(const CLLogger&);
	CLLogger& operator=(const CLLogger&);

	CLLogger();
	~CLLogger();

	int m_Fd;
	static CLLogger *m_pLog;
};

#endif