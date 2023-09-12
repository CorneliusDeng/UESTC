#include <iostream>
#include "CLProcess.h"

using namespace std;

class CLMyProcess : public CLProcess<CLMyProcess>
{
public:
	CLStatus StartFunctionOfProcess(void *pContext)
	{
		cout << (long)pContext << endl;

		return CLStatus(0, 0);
	}
};

int main()
{
    CLMyProcess process;
    process.Run((void *)2);

    return 0;
}
