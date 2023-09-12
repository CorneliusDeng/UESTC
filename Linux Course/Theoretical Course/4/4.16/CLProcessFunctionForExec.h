#ifndef CLProcessFunctionForExec_H
#define CLProcessFunctionForExec_H

#include "CLExecutiveFunctionProvider.h"

class CLProcessFunctionForExec : public CLExecutiveFunctionProvider
{
public:
	CLProcessFunctionForExec();
	virtual ~CLProcessFunctionForExec();

	virtual CLStatus RunExecutiveFunction(void* pCmdLine);

private:
	CLProcessFunctionForExec(const CLProcessFunctionForExec&);
	CLProcessFunctionForExec& operator=(const CLProcessFunctionForExec&);
};

#endif