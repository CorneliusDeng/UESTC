main : CLProcessTester.o test.o CLConditionVariable.o CLCriticalSection.o CLEvent.o CLExecutive.o CLExecutiveCommunication.o CLExecutiveFunctionForMsgLoop.o CLExecutiveFunctionProvider.o CLExecutiveInitialFinishedNotifier.o CLExecutiveNameServer.o CLLogger.o CLMessage.o CLMessageLoopManager.o CLMessageObserver.o CLMessageQueueBySTLqueue.o CLMsgLoopManagerForSTLqueue.o CLMutex.o CLNonThreadForMsgLoop.o CLProcess.o CLProcessFunctionForExec.o CLStatus.o CLThread.o CLThreadCommunicationBySTLqueue.o CLThreadForMsgLoop.o CLThreadInitialFinishedNotifier.o 
	g++ -I../lib/gtest-1.4.0/include -o main CLProcessTester.o test.o CLConditionVariable.o CLCriticalSection.o CLEvent.o CLExecutive.o CLExecutiveCommunication.o CLExecutiveFunctionForMsgLoop.o CLExecutiveFunctionProvider.o CLExecutiveInitialFinishedNotifier.o CLExecutiveNameServer.o CLLogger.o CLMessage.o CLMessageLoopManager.o CLMessageObserver.o CLMessageQueueBySTLqueue.o CLMsgLoopManagerForSTLqueue.o CLMutex.o CLNonThreadForMsgLoop.o CLProcess.o CLProcessFunctionForExec.o CLStatus.o CLThread.o CLThreadCommunicationBySTLqueue.o CLThreadForMsgLoop.o CLThreadInitialFinishedNotifier.o ../lib/gtest-1.4.0/libgtest.a -g -lpthread
	rm *.o

CLProcessTester.o : CLProcessTester.cpp
	g++ -I../lib/gtest-1.4.0/include -o CLProcessTester.o -c CLProcessTester.cpp -I../4.11 -g

test.o : test.cpp
	g++ -I../lib/gtest-1.4.0/include -o test.o -c test.cpp -I../4.11 -g

CLConditionVariable.o : ../4.11/CLConditionVariable.cpp
	g++ -o CLConditionVariable.o -c ../4.11/CLConditionVariable.cpp -I../4.11 -g

CLCriticalSection.o : ../4.11/CLCriticalSection.cpp
	g++ -o CLCriticalSection.o -c ../4.11/CLCriticalSection.cpp -I../4.11 -g

CLEvent.o : ../4.11/CLEvent.cpp
	g++ -o CLEvent.o -c ../4.11/CLEvent.cpp -I../4.11 -g

CLExecutive.o : ../4.11/CLExecutive.cpp
	g++ -o CLExecutive.o -c ../4.11/CLExecutive.cpp -I../4.11 -g

CLExecutiveCommunication.o : ../4.11/CLExecutiveCommunication.cpp
	g++ -o CLExecutiveCommunication.o -c ../4.11/CLExecutiveCommunication.cpp -I../4.11 -g

CLExecutiveFunctionForMsgLoop.o : ../4.11/CLExecutiveFunctionForMsgLoop.cpp
	g++ -o CLExecutiveFunctionForMsgLoop.o -c ../4.11/CLExecutiveFunctionForMsgLoop.cpp -I../4.11 -g

CLExecutiveFunctionProvider.o : ../4.11/CLExecutiveFunctionProvider.cpp
	g++ -o CLExecutiveFunctionProvider.o -c ../4.11/CLExecutiveFunctionProvider.cpp -I../4.11 -g

CLExecutiveInitialFinishedNotifier.o : ../4.11/CLExecutiveInitialFinishedNotifier.cpp
	g++ -o CLExecutiveInitialFinishedNotifier.o -c ../4.11/CLExecutiveInitialFinishedNotifier.cpp -I../4.11 -g

CLExecutiveNameServer.o : ../4.11/CLExecutiveNameServer.cpp
	g++ -o CLExecutiveNameServer.o -c ../4.11/CLExecutiveNameServer.cpp -I../4.11 -g

CLLogger.o : ../4.11/CLLogger.cpp
	g++ -o CLLogger.o -c ../4.11/CLLogger.cpp -I../4.11 -g

CLMessage.o : ../4.11/CLMessage.cpp
	g++ -o CLMessage.o -c ../4.11/CLMessage.cpp -I../4.11 -g

CLMessageLoopManager.o : ../4.11/CLMessageLoopManager.cpp
	g++ -o CLMessageLoopManager.o -c ../4.11/CLMessageLoopManager.cpp -I../4.11 -g

CLMessageObserver.o : ../4.11/CLMessageObserver.cpp
	g++ -o CLMessageObserver.o -c ../4.11/CLMessageObserver.cpp -I../4.11 -g

CLMessageQueueBySTLqueue.o : ../4.11/CLMessageQueueBySTLqueue.cpp
	g++ -o CLMessageQueueBySTLqueue.o -c ../4.11/CLMessageQueueBySTLqueue.cpp -I../4.11 -g

CLMsgLoopManagerForSTLqueue.o : ../4.11/CLMsgLoopManagerForSTLqueue.cpp
	g++ -o CLMsgLoopManagerForSTLqueue.o -c ../4.11/CLMsgLoopManagerForSTLqueue.cpp -I../4.11 -g

CLMutex.o : ../4.11/CLMutex.cpp
	g++ -o CLMutex.o -c ../4.11/CLMutex.cpp -I../4.11 -g

CLNonThreadForMsgLoop.o : ../4.11/CLNonThreadForMsgLoop.cpp
	g++ -o CLNonThreadForMsgLoop.o -c ../4.11/CLNonThreadForMsgLoop.cpp -I../4.11 -g

CLProcess.o : ../4.11/CLProcess.cpp
	g++ -o CLProcess.o -c ../4.11/CLProcess.cpp -I../4.11 -g

CLProcessFunctionForExec.o : ../4.11/CLProcessFunctionForExec.cpp
	g++ -o CLProcessFunctionForExec.o -c ../4.11/CLProcessFunctionForExec.cpp -I../4.11 -g

CLStatus.o : ../4.11/CLStatus.cpp
	g++ -o CLStatus.o -c ../4.11/CLStatus.cpp -I../4.11 -g

CLThread.o : ../4.11/CLThread.cpp
	g++ -o CLThread.o -c ../4.11/CLThread.cpp -I../4.11 -g

CLThreadCommunicationBySTLqueue.o : ../4.11/CLThreadCommunicationBySTLqueue.cpp
	g++ -o CLThreadCommunicationBySTLqueue.o -c ../4.11/CLThreadCommunicationBySTLqueue.cpp -I../4.11 -g

CLThreadForMsgLoop.o : ../4.11/CLThreadForMsgLoop.cpp
	g++ -o CLThreadForMsgLoop.o -c ../4.11/CLThreadForMsgLoop.cpp -I../4.11 -g

CLThreadInitialFinishedNotifier.o : ../4.11/CLThreadInitialFinishedNotifier.cpp
	g++ -o CLThreadInitialFinishedNotifier.o -c ../4.11/CLThreadInitialFinishedNotifier.cpp -I../4.11 -g