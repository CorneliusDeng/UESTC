test: FileOption.o test.o 
	g++ -o test FileOption.o test.o 
FileOption.o : FileOption.h FileOption.cpp
	g++ -c FileOption.h FileOption.cpp
test.o : test.cpp
	g++ -c test.cpp