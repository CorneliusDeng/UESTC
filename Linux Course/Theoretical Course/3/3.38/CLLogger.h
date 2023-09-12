#ifndef CLLogger_H
#define CLLogger_H

#include <pthread.h>
#include "CLStatus.h"

/*
用于向文件LOG_FILE_NAME中，记录日志信息
*/
class CLLogger
{
public:
	static CLLogger* GetInstance();
	static CLStatus WriteLogMsg(const char *pstrMsg, long lErrorCode);
	CLStatus WriteLog(const char *pstrMsg, long lErrorCode);
	CLStatus Flush();

private:
	static void OnProcessExit();

	CLStatus WriteMsgAndErrcodeToFile(const char *pstrMsg, const char *pstrErrcode);

	static pthread_mutex_t *InitializeMutex();

private:
	CLLogger(const CLLogger&);
	CLLogger& operator=(const CLLogger&);

	CLLogger();
	~CLLogger();

private:
	int m_Fd;
	pthread_mutex_t *m_pMutexForWritingLog;
	
	static CLLogger *m_pLog;
	static pthread_mutex_t *m_pMutexForCreatingLogger;

private:
	char *m_pLogBuffer;
	unsigned int m_nUsedBytesForBuffer;

private:
	bool m_bFlagForProcessExit;
};

#endif