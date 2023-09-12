#include <gtest/gtest.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include "CLLogger.h"

TEST(CLLogger, WriteLog_pstrmsg_0)
{
	CLLogger *pLogger = CLLogger::GetInstance();
	if(pLogger != 0)
	{
		CLStatus s = pLogger->WriteLog(0, 0);	
		EXPECT_EQ(s.m_clReturnCode, -1);
	}
}

TEST(CLLogger, WriteLog_pstrmsg_empty)
{
	CLLogger *pLogger = CLLogger::GetInstance();
	if(pLogger != 0)
	{
		CLStatus s = pLogger->WriteLog("", 0);	
		EXPECT_EQ(s.m_clReturnCode, -1);
	}
}

TEST(CLLogger, WriteLogMsg_pstrmsg_0)
{
	CLStatus s = CLLogger::WriteLogMsg(0, 0);
	EXPECT_EQ(s.m_clReturnCode, -1);
}

TEST(CLLogger, WriteLogMsg_pstrmsg_empty)
{
	CLStatus s = CLLogger::WriteLogMsg("", 0);
	EXPECT_EQ(s.m_clReturnCode, -1);
}

TEST(CLLogger, Flush)
{
	CLLogger *pLog = CLLogger::GetInstance();
	EXPECT_TRUE(pLog != 0);

	CLStatus s = pLog->Flush();
	EXPECT_TRUE(s.IsSuccess());
}

TEST(CLLogger, Features)
{
	const int n = 100000;
	for(int i = 0; i < n; i++)
		CLLogger::WriteLogMsg("nihao", 0);

	CLLogger *pLog = CLLogger::GetInstance();
	EXPECT_TRUE(pLog != 0);

	CLStatus s = pLog->Flush();
	EXPECT_TRUE(s.IsSuccess());

	FILE *fp = fopen("CLLogger.txt", "r");

	for(int i = 0; i < n; i++)
	{
		char buf[256];
		fgets(buf, 256, fp);
		EXPECT_EQ(strcmp(buf, "nihao	Error code: 0\r\n"), 0);
	}

	fclose(fp);
}

#define NUM 30

long ObjectsForLog[NUM];

void* TestThreadForCLLog(void *arg)
{	
	long i = (long)arg;	
	long j = (long)CLLogger::GetInstance();	
	ObjectsForLog[i] = j;
}

TEST(CLLogger, Singleton)
{	
	pthread_t tid[NUM];	
	for(long i = 0; i < NUM; i++)	
	{		
		pthread_create(&(tid[i]), 0, TestThreadForCLLog, (void *)i);	
	}	

	for(long i = 0; i < NUM; i++)
	{		
		pthread_join(tid[i], 0);	
	}	

	long j = (long)CLLogger::GetInstance();	
	for(long i = 0; i < NUM; i++)	
	{		
		EXPECT_EQ(j, ObjectsForLog[i]);	
	}
}

void* TestThreadForCLLog1(void *arg)
{
	long i = (long)arg;

	for(int j = 0; j < 1000; j++)
	{
		if((i % 2) == 0)
		{
			CLStatus s = CLLogger::GetInstance()->WriteLog("dddfaefgds", i);
			EXPECT_TRUE(s.IsSuccess());
		}
		else
			CLLogger::GetInstance()->Flush();
	}
}

TEST(CLLogger, WriteLogForMultiThread)
{
	pthread_t tid[NUM];
	for(int i = 0; i < NUM; i++)
	{
		pthread_create(&(tid[i]), 0, TestThreadForCLLog1, (void *)((long)i));
	}

	for(int i = 0; i < NUM; i++)
	{
		pthread_join(tid[i], 0);
	}
}
