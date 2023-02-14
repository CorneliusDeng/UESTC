main: main.c tool.o bar.o
	gcc main.c tool.o bar.o -o main

tool.o: tool.c
	gcc -c tool.c

bar.o: bar.c
	gcc -c bar.c

clean:
	rm *.o main
