#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "CLProcess.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLLogger.h"

CLProcess::CLProcess(CLExecutiveFunctionProvider *pExecutiveFunctionProvider) : CLExecutive(pExecutiveFunctionProvider)
{
	m_bProcessCreated = false;
	m_bWaitForDeath = false;
	m_bExecSuccess = true;
}

CLProcess::CLProcess(CLExecutiveFunctionProvider *pExecutiveFunctionProvider, bool bWaitForDeath) : CLExecutive(pExecutiveFunctionProvider)
{
	m_bProcessCreated = false;
	m_bWaitForDeath = bWaitForDeath;
	m_bExecSuccess = true;
}

CLProcess::~CLProcess()
{
}

CLStatus CLProcess::Run(void *pstrCmdLine)
{
	if(m_bProcessCreated)
		return CLStatus(-1, 0);

	m_ProcessID = vfork();
	if(m_ProcessID == 0)
	{
		m_pExecutiveFunctionProvider->RunExecutiveFunction(pstrCmdLine);

		m_bExecSuccess = false;

		_exit(0);
	}
	else if(m_ProcessID == -1)
	{
		CLLogger::WriteLogMsg("In CLProcess::Run(), vfork error", errno);
		delete this;
		return CLStatus(-1, 0);
	}
	else
	{
		if(!m_bExecSuccess)
		{
			waitpid(m_ProcessID, 0, 0);
			delete this;
			return CLStatus(-1, 0);
		}

		m_bProcessCreated = true;

		if(!m_bWaitForDeath)
			delete this;

		return CLStatus(0, 0);
	}
}

CLStatus CLProcess::WaitForDeath()
{
	if(!m_bWaitForDeath)
		return CLStatus(-1, 0);

	if(!m_bProcessCreated)
		return CLStatus(-1, 0);

	if(waitpid(m_ProcessID, 0, 0) == -1)
	{
		CLLogger::WriteLogMsg("In CLProcess::WaitForDeath(), waitpid error", errno);
		return CLStatus(-1, errno);
	}

	delete this;

	return CLStatus(0, 0);
}