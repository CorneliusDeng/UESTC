#include <iostream>
#include <unistd.h>
#include <fcntl.h>

using namespace std;

int main()
{
	int fd = open("a.txt", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
	if(fd == -1)
	{
		cout << "open error" << endl;
		return 0;
	}

	struct flock lock;
	lock.l_type = F_RDLCK;
	lock.l_start = 0;
	lock.l_whence = SEEK_SET;
	lock.l_len = 0;

	if(fcntl(fd, F_SETLKW, &lock) == -1)
	{
		cout << "fcntl error" << endl;
		close(fd);
		return 0;
	}

	cout << "child got reading lock" << endl;

	lock.l_type = F_UNLCK;
	if(fcntl(fd, F_SETLKW, &lock) == -1)
	{
		cout << "fcntl error" << endl;
		close(fd);
		return 0;
	}

	cout << "child unlock" << endl;

	close(fd);
	return 0;
}
