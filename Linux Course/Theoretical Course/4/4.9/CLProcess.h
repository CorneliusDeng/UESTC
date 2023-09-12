#ifndef CLProcess_H
#define CLProcess_H

#include <unistd.h>
#include "CLStatus.h"
#include "CLLogger.h"
#include <stdlib.h>
#include <errno.h>

template<typename TProcessFunctionProvider>
class CLProcess
{
public:
	CLProcess()
	{
	}

	~CLProcess()
	{
	}

	CLStatus Run(void *pContext)
	{
		m_ProcessID = fork();
		if(m_ProcessID == 0)
		{
			m_ProcessID = getpid();

			TProcessFunctionProvider *pT = (TProcessFunctionProvider *)(this);

			pT->StartFunctionOfProcess(pContext);

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

private:
	CLProcess(const CLProcess&);
	CLProcess& operator=(const CLProcess&);

protected:
	pid_t m_ProcessID;
};

#endif
