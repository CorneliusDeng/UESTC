#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include "LibExecutive.h"
#include "CLProcess.h"
#include "CLProcessFunctionForExec.h"

using namespace std;

int main()
{
	int fd = -1;

	try
	{
		if(!CLLibExecutiveInitializer::Initialize().IsSuccess())
		{
			cout << "Initialize error" << endl;
			return 0;
		}

		fd = open("./test/a.txt", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
		if(fd == -1)
		{
			cout << "open error" << endl;
			throw CLStatus(-1, 0);
		}

		struct flock lock;
		lock.l_type = F_WRLCK;
		lock.l_start = 0;
		lock.l_whence = SEEK_SET;
		lock.l_len = 0;

		if(fcntl(fd, F_SETLKW, &lock) == -1)
		{
			cout << "fcntl error" << endl;
			throw CLStatus(-1, 0);
		}

		CLExecutive *p = new CLProcess(new CLProcessFunctionForExec, true);
		if(!p->Run((void *)"./test/a.out").IsSuccess())
		{
			cout << "Run error" << endl;
			throw CLStatus(-1, 0);
		}

		p->WaitForDeath();

		throw CLStatus(0, 0);

	}
	catch(CLStatus& s)
	{
		if(fd != -1)
			close(fd);

		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}