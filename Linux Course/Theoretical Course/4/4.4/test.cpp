#include<iostream>
#include<unistd.h>
#include<sys/wait.h>

using namespace std;

int main()
{
  pid_t pid = fork();
  if(pid == 0)
  {
    cout << "pid = " << getpid() << endl;
    execl("./a.out", "./a.out", "Hello ", "World!", (char *)0);
    cout << "back" << endl;
  }

  wait(NULL);
  return 0;
}