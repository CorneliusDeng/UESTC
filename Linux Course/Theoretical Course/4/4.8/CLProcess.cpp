#include <errno.h>
#include <stdlib.h>
#include <iostream>
#include <sys/wait.h>
#include "CLExecutiveFunctionProvider.h"
#include "CLProcess.h"
#include "CLLogger.h"

using namespace std;

CLProcess::CLProcess(CLExecutiveFunctionProvider *pExecutiveFunctionProvider) : CLExecutive(pExecutiveFunctionProvider)
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

		m_pExecutiveFunctionProvider->RunExecutiveFunction(pContext);

		delete m_pExecutiveFunctionProvider;

		delete this;

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

CLStatus CLProcess::WaitForDeath()
{
	if(m_ProcessID == -1)
		return CLStatus(-1, 0);

	if(waitpid(m_ProcessID, 0, 0) == -1)
		return CLStatus(-1, errno);
	else
		return CLStatus(0, 0);
}