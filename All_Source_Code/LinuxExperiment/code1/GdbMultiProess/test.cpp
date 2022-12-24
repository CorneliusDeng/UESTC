#include<iostream>
#include<sys/types.h>
#include<unistd.h>
#include<stdlib.h>

using namespace std;

typedef void (*PROCESS)(void *pContext);

pid_t CreateProcess(PROCESS child, void *pContext)
{
  pid_t pid = fork();
  
  if(0 == pid)
  {
    sleep(60);
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
  long long i = (long long)pContext;
  cout << "In child: " << i << endl;
}

int main()
{
  int i = 90;

  CreateProcess(MyChild, (void *)i);
 
  cout << "In Father" << endl;

  return 0;
}
