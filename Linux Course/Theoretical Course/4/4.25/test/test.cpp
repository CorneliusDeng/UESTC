#include <iostream>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include "LibExecutive.h"

using namespace std;

int main()
{
	int *paddr = (int *)-1;
	int shmid = -1;

	try
	{
		key_t key = ftok("/tmp/a.txt", 0);
		if(key == -1)
		{
			cout << "ftok error" << endl;
			throw CLStatus(-1, 0);
		}

		shmid = shmget(key, 0, 0);
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

		cout << "in child: " << *paddr << endl;

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

		return 0;
	}
}