#ifndef CLEXECUTIVE_H
#define CLEXECUTIVE_H

#include "CLStatus.h"

class CLCoordinator;

/* 
CLExecutive类用于创建一个执行体
*/
class CLExecutive
{
public:
	explicit CLExecutive(CLCoordinator *pCoordinator);
	virtual ~CLExecutive();

	virtual CLStatus Run() = 0;

	virtual CLStatus WaitForDeath() = 0;
	
protected:
	CLCoordinator *m_pCoordinator;

private:
	CLExecutive(const CLExecutive&);
	CLExecutive& operator=(const CLExecutive&);
};

#endif
