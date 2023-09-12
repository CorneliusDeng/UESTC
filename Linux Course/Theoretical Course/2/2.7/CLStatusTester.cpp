#include <iostream>
#include "CLStatus.h"

using namespace std;

CLStatus f()
{
	CLStatus s(-1, 2);
	return s;
}

int main()
{
	CLStatus s = f();
	if(!s.IsSuccess())
	{
		cout << "f error" << endl;
		cout << "error code: " << s.m_clErrorCode << endl;
	}
	return 0;
}