#include "CLRegularCoordinator.h"
#include "CLExecutiveFunctionProvider.h"
#include "CLExecutive.h"

CLRegularCoordinator::CLRegularCoordinator()
{
}

CLRegularCoordinator::~CLRegularCoordinator()
{
}

CLStatus CLRegularCoordinator::Run(void *pContext)
{
	if((m_pExecutive == 0) || (m_pExecutiveFunctionProvider == 0))
		return CLStatus(-1, 0);

	m_pContext = pContext;

	return m_pExecutive->Run();
}

CLStatus CLRegularCoordinator::ReturnControlRights()
{
	return m_pExecutiveFunctionProvider->RunExecutiveFunction(m_pContext);
}

CLStatus CLRegularCoordinator::WaitForDeath()
{
	return m_pExecutive->WaitForDeath();
}