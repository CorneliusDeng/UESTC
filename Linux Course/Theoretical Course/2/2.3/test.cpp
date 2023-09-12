#include <fcntl.h>
#include <iostream>
#include <errno.h>
#include <string.h>

using namespace std;

int main()
{
  int fd = open("a.txt", O_RDONLY);
  if(-1 == fd)  
  { 
    cout << "open error" << endl;
    cout << "errno is " << errno << endl;
    cout << strerror(errno) << endl;
  }

  return 0;
}
