#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>

int main(){
	pid_t id = fork();
	if (id < 0){
		// failure, father return -1
		perror("fork error\n");
		return -1;
	}
	else if (id == 0){
		// success, child process return 0
		printf("Here is child, id:%d, father id:%d \n", getpid(), getppid());
	}
	else{
		// success, father process return child id
		sleep(1);
		printf("Here is father, id:%d\n", getpid());
	}
	return 0;
}
