#include<iostream>
#include<unistd.h>

using namespace std;

int main(int argc, char *argv[])
{
  cout << "in new program pid = " << getpid() << endl;
  cout << argv[1] << argv[2] << endl;

  return 0;
}