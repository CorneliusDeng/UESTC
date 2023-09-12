#include<iostream>
#include<unistd.h>

using namespace std;

int  main(void)
{
	int i = 0;
	pid_t pid = fork();

	if(pid == -1)
	{
		cout << "fork error" << endl;
		return 0;
	}
	else if(pid == 0)
	{
		cout << "child id : " << getpid() << endl;
		i = 5;
	}
	else
	{
		sleep(2);
		cout << "father id : " << getpid() << endl;
	}

	cout << "id " << getpid() << ": i = " << i << endl;
	
	return 0;
}
