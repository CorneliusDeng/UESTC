#include <fcntl.h>
#include <iostream>

using namespace std;

int main()
{
  int fd = open("a.txt", O_RDONLY);
  if(-1 == fd)  
  { 
    cout << "open error" << endl;
  }

  return 0;
}
