#include <string.h>
#include <vector>
#include <unistd.h>
#include <errno.h>
#include "CLProcessFunctionForExec.h"
#include "CLLogger.h"

using namespace std;

CLProcessFunctionForExec::CLProcessFunctionForExec()
{
}

CLProcessFunctionForExec::~CLProcessFunctionForExec()
{
}

CLStatus CLProcessFunctionForExec::RunExecutiveFunction(void* pCmdLine)
{
	if(pCmdLine == 0)
		return CLStatus(-1, 0);

	int len = strlen((char *)pCmdLine);
	if(len == 0)
		return CLStatus(-1, 0);

	char *pstrCmdLine = new char[len + 1];
	strcpy(pstrCmdLine, (char *)pCmdLine);

	char *p = pstrCmdLine;

	vector<char *> vstrArgs;

	while(char *q = strsep(&p, " "))
	{
		if(*q == 0)
			continue;

		vstrArgs.push_back(q);
	}

	char **argv = new char* [vstrArgs.size() + 1];
	for(int i = 0; i < vstrArgs.size(); i++)
		argv[i] = vstrArgs[i];
	argv[vstrArgs.size()] = NULL;

	execv(argv[0], argv);

	CLLogger::WriteLogDirectly("In CLProcessFunctionForExec::RunExecutiveFunction(), execv error", errno);

	delete [] argv;

	delete [] pstrCmdLine;
	
	return CLStatus(-1, 0);
}