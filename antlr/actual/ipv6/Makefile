all: nogoto.c test.c rte_lpm6.c rte_lpm6.h ipv6.c ipv6.h
	gcc -O3 -o nogoto nogoto.c rte_lpm6.c ipv6.c -lnuma -Wall -Werror -Wno-unused-result -lpapi
	gcc -O3 -o test test.c rte_lpm6.c -lnuma -Wall -Werror -Wno-unused-result
clean:
	rm *.o test
