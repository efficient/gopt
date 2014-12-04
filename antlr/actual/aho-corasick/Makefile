all:
	gcc -O3 -o handopt aho.c ds_queue.c handopt.c util.c -lpapi -Wno-unused-result -lrt -lpthread -Wall -Werror -march=native
	gcc -O3 -o noopt aho.c ds_queue.c noopt.c util.c -lpapi -Wno-unused-result -lrt -lpthread -Wall -Werror
clean:
	rm handopt noopt
