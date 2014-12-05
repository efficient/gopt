all:
	gcc -O3 -o nogoto nogoto.c city.c util.c ndn.c -lrt -lpapi -Wall -Werror -Wno-unused-result
	gcc -O3 -o goto goto.c city.c util.c ndn.c -lrt -lpapi -Wall -Werror -Wno-unused-result -Wno-unused-label

clean:
	rm -f *.o goto nogoto
