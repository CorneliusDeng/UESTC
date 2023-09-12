#include <unistd.h>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    cout << "in child main" << endl;

    if(write(3, "nihao", 5) == -1)
	cout << "child write error" << endl;
    else
	cout << "child write success" << endl;

    return 0;
}

