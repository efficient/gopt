all:
	gcc -O3 -o nogoto aho.c ds_queue.c nogoto.c util.c -lpapi -Wno-unused-result
	gcc -O3 -o goto aho.c ds_queue.c goto.c util.c -lpapi -Wno-unused-result
clean:
	rm nogoto goto
