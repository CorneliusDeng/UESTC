#include <iostream>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include "LibExecutive.h"
#include "CLProcess.h"
#include "CLProcessFunctionForExec.h"

using namespace std;

int main()
{
	int *paddr = (int *)-1;
	int shmid = -1;

	try
	{
		if(!CLLibExecutiveInitializer::Initialize().IsSuccess())
		{
			cout << "Initialize error" << endl;
			return 0;
		}
		
		int fd = open("/tmp/a.txt", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
		if(fd == -1)
		{
			cout << "open error" << endl;
			throw CLStatus(-1, 0);
		}

		close(fd);

		key_t key = ftok("/tmp/a.txt", 0);
		if(key == -1)
		{
			cout << "ftok error" << endl;
			throw CLStatus(-1, 0);
		}

		shmid = shmget(key, 4, IPC_CREAT);
		if(shmid == -1)
		{
			cout << "shmget error" << endl;
			throw CLStatus(-1, 0);
		}

		paddr = (int *)shmat(shmid, 0, 0);
		if(paddr == (int *)-1)
		{
			cout << "shmat error" << endl;
			throw CLStatus(-1, 0);
		}

		*paddr = 5;

		CLExecutive *process = new CLProcess(new CLProcessFunctionForExec, true);
		if(!(process->Run((void *)"./test/a.out")).IsSuccess())
			cout << "process Run error" << endl;
		else
			process->WaitForDeath();

		throw CLStatus(0, 0);
	}
	catch(CLStatus& s)
	{
		if(paddr != (int *)-1)
		{
			shmdt(paddr);
		}

		if(shmid != -1)
		{
			shmctl(shmid, IPC_RMID, 0);
		}

		if(!CLLibExecutiveInitializer::Destroy().IsSuccess())
			cout << "Destroy error" << endl;

		return 0;
	}
}