#include "CLStatus.h"

CLStatus::CLStatus(long lReturnCode, long lErrorCode)
{
	m_lReturnCode = lReturnCode;
	m_lErrorCode = lErrorCode;
}

CLStatus::~CLStatus()
{
}

bool CLStatus::IsSuccess()
{
	if(m_lReturnCode >= 0)
		return true;
	else
		return false;
}

long CLStatus::GetErrorCode()
{
	return m_lErrorCode;
}