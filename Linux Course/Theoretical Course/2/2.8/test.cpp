#include <iostream>

using namespace std;

class A
{
public:
	A()
	{
		cout << "In A(): " << hex << (long)this << endl;
	}

	A(const A&)
	{
		cout << "In A(const A&): " << hex << (long)this << endl;
	}

	~A()
	{
		cout << "In ~A(): " << hex << (long)this << endl;
	}

	A& operator=(const A& a)
	{

		cout << "In operator=: " << hex << (long)this << " = " << hex << (long)(&a) << endl;
		return *this;
	}
};

A f()
{
	//A a;
	return A();
}

int main(int argc, char* argv[])
{
	A a = f();
	return 0;
}