#include "CLProcess.h"

int main()
{
    CLProcess process;
    process.Run((void *)2);
	process.WaitForDeath();

    return 0;
}
