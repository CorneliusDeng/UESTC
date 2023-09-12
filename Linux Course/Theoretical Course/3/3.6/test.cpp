#include <iostream>
#include <unistd.h>
#include "CLThread.h"

using namespace std;

int main()
{
    CLThread thread;
    thread.Run((void *)2);
    thread.WaitForDeath();
    return 0;
}
