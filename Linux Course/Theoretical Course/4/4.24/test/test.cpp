#include <iostream>
#include <fcntl.h>
#include "LibExecutive.h"
#include "CLProcess.h"
#include "CLProcessFunctionForExec.h"

using namespace std;

pthread_mutex_t pmutex = PTHREAD_MUTEX_INITIALIZER;

void ReadAndWriteFile(int fd)
{
	for(int i = 0; i < 1000000; i++)
	{
		CLMutex mutex("test_for_mutex", &pmutex);
		CLCriticalSection cs(&mutex);

		long k = 0;

		lseek(fd, SEEK_SET, 0);
		read(fd, &k, sizeof(long));

		k++;

		lseek(fd, SEEK_SET, 0);
		write(fd, &k, sizeof(long));
	}
}

class CLThreadFunc : public CLExecutiveFunctionProvider
{
public:
	virtual CLStatus RunExecutiveFunction(void* pContext)
	{
		long fd = (long)pContext;

		ReadAndWriteFile((int)fd);

		return CLStatus(0, 0);
	}
};

int main()
{
	int fd = -1;
	CLExecutive *pthread = 0;

	try
	{
		if(!CLLibExecutiveInitializer::Initialize().IsSuccess())
		{
			cout << "Initialize error" << endl;
			return 0;
		}
		
		fd = open("a.txt", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
		if(fd == -1)
		{
			cout << "open error" << endl;
			throw CLStatus(-1, 0);
		}

		pthread = new CLThread(new CLThreadFunc, true);
		if(!(pthread->Run((void *)((long)fd))).IsSuccess())
		{
			cout << "thread Run error" << endl;
			pthread = 0;
			throw CLStatus(-1, 0);
		}

		ReadAndWriteFile(fd);

		throw CLStatus(0, 0);

	}
	catch(CLStatus& s)
	{
		if(pthread != 0)
			pthread->WaitForDeath();

		if(fd != -1)
			if(close(fd) == -1)
				cout << "close error" << endl;

		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}