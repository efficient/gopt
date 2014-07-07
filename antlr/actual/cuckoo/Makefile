CFLAGS	:= -O3 -Wall -Werror

all:
	gcc -O3 -o nogoto nogoto.c city.c -lrt
	gcc -O3 -o goto goto.c city.c -lrt
	gcc -O3 -o handopt handopt.c city.c -lrt
	gcc -O3 -o citybench citybench.c city.c -lrt

clean:
	rm -f *.o goto nogoto
