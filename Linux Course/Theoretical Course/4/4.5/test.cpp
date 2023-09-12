#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

using namespace std;

typedef void (*PROCESS)(void *pContext);

pid_t CreateProcess(PROCESS child, void *pContext)
{
  pid_t pid = fork();
  
  if(0 == pid)
  {
    child(pContext);
    exit(0);
  }
  else
  {
    return pid;
  }
}

void MyChild(void *pContext)
{
  long i = (long)pContext;
  cout << "In child: " << i << endl;
}

int main()
{
  long i = 90;

  CreateProcess(MyChild, (void *)i);
 
  cout << "In Father" << endl;

  return 0;
}
