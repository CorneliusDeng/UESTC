#include <iostream>
#include <string.h>
using namespace std;

int main(int argc, char **argv)
{
	if(argc != 3)
		throw "error number";

	if(strcmp(argv[0], "./b.out") != 0)
		throw "error b.out";

	if(strcmp(argv[1], "hello") != 0)
		throw "error2 hello ";

	if(strcmp(argv[2], "world") != 0)
		throw "error world";
	return 0;
}