#ifndef CLProcess_H
#define CLProcess_H

#include <unistd.h>
#include "CLStatus.h"
#include "CLExecutive.h"

class CLProcess : public CLExecutive
{
public:
	explicit CLProcess(CLExecutiveFunctionProvider *pExecutiveFunctionProvider);
	virtual ~CLProcess();

	virtual CLStatus Run(void *pContext = 0);
	virtual CLStatus WaitForDeath();

private:
	CLProcess(const CLProcess&);
	CLProcess& operator=(const CLProcess&);

protected:
	pid_t m_ProcessID;
};

#endif