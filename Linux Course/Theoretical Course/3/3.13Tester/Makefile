main : test.o CLLoggerTester.o CLStatus.o CLLogger.o 
	g++ -I../lib/gtest-1.4.0/include -o main test.o CLLoggerTester.o CLStatus.o CLLogger.o ../lib/gtest-1.4.0/libgtest.a -g -lpthread
	rm *.o

test.o : test.cpp
	g++ -I../lib/gtest-1.4.0/include -o test.o -c test.cpp -g

CLLoggerTester.o : CLLoggerTester.cpp
	g++ -I../lib/gtest-1.4.0/include -I../3.13 -o CLLoggerTester.o -c CLLoggerTester.cpp -g

CLStatus.o : ../3.13/CLStatus.cpp
	g++ -o CLStatus.o -c ../3.13/CLStatus.cpp -g

CLLogger.o : ../3.13/CLLogger.cpp
	g++ -o CLLogger.o -c ../3.13/CLLogger.cpp -g

