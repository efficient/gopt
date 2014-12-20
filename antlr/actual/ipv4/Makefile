all: test.c rte_lpm.c rte_lpm.h
	gcc -O3 -o test test.c rte_lpm.c -lnuma -Wall -Werror -Wno-unused-result
clean:
	rm *.o test
