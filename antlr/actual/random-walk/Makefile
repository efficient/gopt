all:
	gcc -O3 -o nogoto nogoto.c rand-walk.c -lrt -Wall -Werror
	gcc -O3 -o goto goto.c rand-walk.c -lrt -Wall -Werror

clean:
	rm -f *.o goto nogoto
