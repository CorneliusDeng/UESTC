#include<pthread.h>
#include<iostream>
#include<unistd.h>

using namespace std;

void *thread(void *arg)
{
    long long i = (long long)arg;
    cout << "in thread, tid = " << pthread_self() << endl;
    cout << "arg is " << i << endl;

    return (void *)0;
}

int main()
{
    pthread_t tid;
    if(pthread_create(&tid, NULL, thread, (void *)2) != 0)
    {
	cout << "pthread_create error" << endl;
	return 0;
    }
    sleep(500);
    cout << "in main thread, tid = " << pthread_self() << endl;
    return 0;
}
