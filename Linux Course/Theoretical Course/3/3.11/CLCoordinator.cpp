#include "CLCoordinator.h"

CLCoordinator::CLCoordinator()
{
}

CLCoordinator::~CLCoordinator()
{
}

void CLCoordinator::SetExecObjects(CLExecutive *pExecutive, CLExecutiveFunctionProvider *pProvider)
{
	m_pExecutive = pExecutive;
	m_pExecutiveFunctionProvider = pProvider;
}

