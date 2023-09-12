test : test.o f.o
	g++ -o test test.o f.o
	rm *.o

test.o : test.cpp
	g++ -c test.cpp -o test.o

f.o : f.cpp
	g++ -c f.cpp -o f.o
