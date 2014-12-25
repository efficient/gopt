all: test.c real-world.c rte_lpm.c rte_lpm.h ipv4.c
	gcc -O3 -o test test.c rte_lpm.c ipv4.c -lnuma -lpapi -Wall -Werror -Wno-unused-result
	gcc -O3 -o real-world real-world.c rte_lpm.c ipv4.c -lpapi -lnuma -Wall -Werror -Wno-unused-result
clean:
	rm *.o test real-world
