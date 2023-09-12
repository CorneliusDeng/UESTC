#ifndef CLProcess_H
#define CLProcess_H

#include <unistd.h>
#include "CLStatus.h"

class CLProcess
{
public:
	CLProcess();
	~CLProcess();

	CLStatus Run(void *pContext);
	CLStatus WaitForDeath();

private:
	CLStatus StartFunctionOfProcess(void *pContext);

private:
	CLProcess(const CLProcess&);
	CLProcess& operator=(const CLProcess&);

private:
	pid_t m_ProcessID;
};

#endif
