all:
	gcc -O3 -o nogoto aho.c ds_queue.c nogoto.c util.c -lpapi -Wno-unused-result -lrt
	gcc -O3 -o handopt aho.c ds_queue.c handopt.c util.c -lpapi -Wno-unused-result -lrt -lpthread
clean:
	rm nogoto goto handopt
