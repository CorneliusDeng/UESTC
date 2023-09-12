#include <iostream>
#include "CLProcess.h"

using namespace std;

class CLMyProcess : public CLProcess
{
public:
	virtual CLStatus StartFunctionOfProcess(void *pContext)
	{
		cout << (long)pContext << endl;

		return CLStatus(0, 0);
	}
};

int main()
{
    CLMyProcess process;
    process.Run((void *)2);
	process.WaitForDeath();

    return 0;
}
