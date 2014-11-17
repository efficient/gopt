all:
	gcc -O3 -o nogoto aho.c ds_queue.c nogoto.c util.c -lpapi -Wno-unused-result -lrt -Wall -Werror
	gcc -O3 -o handopt aho.c ds_queue.c handopt.c util.c -lpapi -Wno-unused-result -lrt -lpthread -Wall -Werror
clean:
	rm nogoto goto handopt
