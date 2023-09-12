#ifndef CLProcess_H
#define CLProcess_H

#include <unistd.h>
#include "CLStatus.h"

class CLProcess
{
public:
	CLProcess();
	virtual ~CLProcess();

	virtual CLStatus Run(void *pContext);
	virtual CLStatus WaitForDeath();

protected:
	virtual CLStatus StartFunctionOfProcess(void *pContext) = 0;

private:
	CLProcess(const CLProcess&);
	CLProcess& operator=(const CLProcess&);

protected:
	pid_t m_ProcessID;
};

#endif
