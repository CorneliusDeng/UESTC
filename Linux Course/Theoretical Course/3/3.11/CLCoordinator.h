#ifndef CLCoordinator_H
#define CLCoordinator_H

#include "CLStatus.h"

class CLExecutive;
class CLExecutiveFunctionProvider;

class CLCoordinator
{
public:
	CLCoordinator();
	virtual ~CLCoordinator();

	void SetExecObjects(CLExecutive *pExecutive, CLExecutiveFunctionProvider *pProvider);

	virtual CLStatus Run(void *pContext) = 0;
	virtual CLStatus ReturnControlRights() = 0;
	virtual CLStatus WaitForDeath() = 0;

private:
	CLCoordinator(const CLCoordinator&);
	CLCoordinator& operator=(const CLCoordinator&);

protected:
	CLExecutive *m_pExecutive;
	CLExecutiveFunctionProvider *m_pExecutiveFunctionProvider;
};

#endif