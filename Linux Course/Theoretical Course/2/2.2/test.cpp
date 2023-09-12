#include <fcntl.h>
#include <iostream>
#include <errno.h>

using namespace std;

int main()
{
  int fd = open("a.txt", O_RDONLY);
  if(-1 == fd)  
  { 
    cout << "open error" << endl;
    cout << "errno is " << errno << endl;
  }

  return 0;
}
