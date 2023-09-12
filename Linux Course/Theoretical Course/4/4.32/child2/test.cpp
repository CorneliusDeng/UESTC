#include <iostream>
#include <fcntl.h>
#include <signal.h>
#include "LibExecutive.h"

using namespace std;

int main()
{
	try
	{
		if(!CLLibExecutiveInitializer::Initialize().IsSuccess())
		{
			cout << "Initialize error" << endl;
			return 0;
		}

		signal(SIGPIPE, SIG_IGN);

		sleep(5);

		int fd = open("/root/4.32/test_pipe", O_WRONLY);
		if(fd == -1)
		{
			cout << "open error" << endl;
			throw CLStatus(-1, 0);
		}

		int length = 40;
		char *pBuf = new char[length];

		cout << "child2 wants to write " << length << endl;
		int WritedLength = write(fd, pBuf, length);
		if(WritedLength == -1)
			cout << "write error" << endl;
		else
			cout << "child2 writed " << WritedLength << endl;

		delete [] pBuf;

		close(fd);

		throw CLStatus(0, 0);
	}
	catch(CLStatus& s)
	{
		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}