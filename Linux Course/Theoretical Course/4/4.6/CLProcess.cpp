#include "CLProcess.h"
#include "CLLogger.h"
#include <errno.h>
#include <stdlib.h>
#include <iostream>
#include <sys/wait.h>

using namespace std;

CLProcess::CLProcess()
{
}

CLProcess::~CLProcess()
{
}

CLStatus CLProcess::Run(void *pContext)
{
	m_ProcessID = fork();
	if(m_ProcessID == 0)
	{
		m_ProcessID = getpid();

		StartFunctionOfProcess(pContext);

		exit(0);
	}
	else if(m_ProcessID == -1)
	{
		CLLogger::WriteLogMsg("In CLProcess::Run(), fork error", errno);
		return CLStatus(-1, 0);
	}
	else
		return CLStatus(0, 0);
}

CLStatus CLProcess::StartFunctionOfProcess(void *pContext)
{
	long i = (long)pContext;

	cout << i << endl;

	return CLStatus(0, 0);
}

CLStatus CLProcess::WaitForDeath()
{
	if(m_ProcessID == -1)
		return CLStatus(-1, 0);

	if(waitpid(m_ProcessID, 0, 0) == -1)
		return CLStatus(-1, errno);
	else
		return CLStatus(0, 0);
}