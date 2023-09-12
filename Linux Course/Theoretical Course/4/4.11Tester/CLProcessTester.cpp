#include <gtest/gtest.h>
#include <iostream>
#include "CLProcess.h"
#include "CLProcessFunctionForExec.h"

using namespace std;

bool g_bForDestroyForProcess = false;
bool g_bForDestroyForProcessFunctionForExec = false;

class CLTestProcess : public CLProcess
{
public:
	CLTestProcess(CLExecutiveFunctionProvider *p) : CLProcess(p)
	{
	}

	CLTestProcess(CLExecutiveFunctionProvider *p, bool bWaitForDeath) : CLProcess(p, bWaitForDeath)
	{
	}

	virtual ~CLTestProcess()
	{
		g_bForDestroyForProcess = true;
	}
};

class CLTestForCLProcessFunctionForExec : public CLProcessFunctionForExec
{
public:
	virtual ~CLTestForCLProcessFunctionForExec()
	{
		g_bForDestroyForProcessFunctionForExec = true;
	}
};

TEST(CLProcess, Param1)
{
	EXPECT_FALSE(g_bForDestroyForProcess);
	EXPECT_FALSE(g_bForDestroyForProcessFunctionForExec);

	CLProcess *p = new CLTestProcess(new CLTestForCLProcessFunctionForExec);
	
	CLStatus s	 = p->Run(0);
	EXPECT_FALSE(s.IsSuccess());

	EXPECT_TRUE(g_bForDestroyForProcess);
	EXPECT_TRUE(g_bForDestroyForProcessFunctionForExec);

	g_bForDestroyForProcess = false;
	g_bForDestroyForProcessFunctionForExec = false;
}

TEST(CLProcess, Param2)
{
	EXPECT_FALSE(g_bForDestroyForProcess);
	EXPECT_FALSE(g_bForDestroyForProcessFunctionForExec);

	CLProcess *p = new CLTestProcess(new CLTestForCLProcessFunctionForExec);

	CLStatus s = p->Run((void *)"");
	EXPECT_FALSE(s.IsSuccess());

	EXPECT_TRUE(g_bForDestroyForProcess);
	EXPECT_TRUE(g_bForDestroyForProcessFunctionForExec);

	g_bForDestroyForProcess = false;
	g_bForDestroyForProcessFunctionForExec = false;
}

TEST(CLProcess, Param3)
{
	EXPECT_FALSE(g_bForDestroyForProcess);
	EXPECT_FALSE(g_bForDestroyForProcessFunctionForExec);

	CLProcess *p = new CLTestProcess(new CLTestForCLProcessFunctionForExec, false);

	CLStatus s = p->Run((void *)"xyz");
	EXPECT_FALSE(s.IsSuccess());

	EXPECT_TRUE(g_bForDestroyForProcess);
	EXPECT_TRUE(g_bForDestroyForProcessFunctionForExec);

	g_bForDestroyForProcess = false;
	g_bForDestroyForProcessFunctionForExec = false;
}

TEST(CLProcess, Normal)
{
	EXPECT_FALSE(g_bForDestroyForProcess);
	EXPECT_FALSE(g_bForDestroyForProcessFunctionForExec);

	CLProcess *p = new CLTestProcess(new CLTestForCLProcessFunctionForExec, false);

	CLStatus s = p->Run((void *)"./b.out hello world");
	EXPECT_TRUE(s.IsSuccess());

	EXPECT_TRUE(g_bForDestroyForProcess);
	EXPECT_TRUE(g_bForDestroyForProcessFunctionForExec);

	g_bForDestroyForProcess = false;
	g_bForDestroyForProcessFunctionForExec = false;
}

TEST(CLProcess, Normal2)
{
	EXPECT_FALSE(g_bForDestroyForProcess);
	EXPECT_FALSE(g_bForDestroyForProcessFunctionForExec);

	CLProcess *p = new CLTestProcess(new CLTestForCLProcessFunctionForExec, false);

	EXPECT_FALSE(p->WaitForDeath().IsSuccess());

	CLStatus s = p->Run((void *)" ./b.out   hello  world  ");
	EXPECT_TRUE(s.IsSuccess());

	EXPECT_TRUE(g_bForDestroyForProcess);
	EXPECT_TRUE(g_bForDestroyForProcessFunctionForExec);

	g_bForDestroyForProcess = false;
	g_bForDestroyForProcessFunctionForExec = false;
}

TEST(CLProcess, Normal3)
{
	EXPECT_FALSE(g_bForDestroyForProcess);
	EXPECT_FALSE(g_bForDestroyForProcessFunctionForExec);

	CLProcess *p = new CLTestProcess(new CLTestForCLProcessFunctionForExec, true);

	EXPECT_FALSE(p->WaitForDeath().IsSuccess());

	CLStatus s = p->Run((void *)"./b.out hello world");
	EXPECT_TRUE(s.IsSuccess());

	CLStatus s1 = p->Run((void *)"./b.out hello world");
	EXPECT_FALSE(s1.IsSuccess());

	EXPECT_TRUE(p->WaitForDeath().IsSuccess());

	EXPECT_TRUE(g_bForDestroyForProcess);
	EXPECT_TRUE(g_bForDestroyForProcessFunctionForExec);

	g_bForDestroyForProcess = false;
	g_bForDestroyForProcessFunctionForExec = false;
}