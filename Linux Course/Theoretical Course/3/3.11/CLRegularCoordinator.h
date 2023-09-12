#ifndef CLRegularCoordinator_H
#define CLRegularCoordinator_H

#include "CLCoordinator.h"

class CLRegularCoordinator : public CLCoordinator
{
public:
	CLRegularCoordinator();
	virtual ~CLRegularCoordinator();

	virtual CLStatus Run(void *pContext);
	virtual CLStatus ReturnControlRights();
	virtual CLStatus WaitForDeath();

private:
	CLRegularCoordinator(const CLRegularCoordinator&);
	CLRegularCoordinator& operator=(const CLRegularCoordinator&);

private:
	void *m_pContext;
};

#endif