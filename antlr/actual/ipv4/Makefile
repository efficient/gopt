all: test.c rte_lpm.c rte_lpm.h ipv4.c
	gcc -O3 -o test test.c rte_lpm.c ipv4.c -lnuma -Wall -Werror -Wno-unused-result
clean:
	rm *.o test
