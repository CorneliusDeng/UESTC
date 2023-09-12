#include <iostream>
#include <unistd.h>
#include "CLThread.h"
#include "CLProcess.h"
#include "CLExecutiveFunctionProvider.h"

using namespace std;

class CLParaPrinter : public CLExecutiveFunctionProvider
{
public:
    CLParaPrinter()
    {
    }

    virtual ~CLParaPrinter()
    {
    }

    virtual CLStatus RunExecutiveFunction(void *pContext)
    {
		long i = (long)pContext;
		cout << i << endl;

		return CLStatus(0, 0);
    }
};

int main()
{
    CLExecutiveFunctionProvider *printer = new CLParaPrinter();
    CLExecutive *pExecutive = new CLProcess(printer);

    pExecutive->Run((void *)2);
    pExecutive->WaitForDeath();

    delete pExecutive;
    delete printer;

    return 0;
}
