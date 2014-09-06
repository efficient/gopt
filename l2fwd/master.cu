#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

// nvcc assumes that all header files are C++ files. Tell it
// that these are C header files
extern "C" {
#include "worker-master.h"
#include "util.h"
}

// Re-define RTE_MAX_LCORE because we don't want to include
// any DPDK header files here
#define WM_MAX_LCORE 64

volatile struct wm_queue *wmq;

int main(int argc, char **argv)
{
	int c;
	int lcore_mask = -1;
	while ((c = getopt (argc, argv, "c:")) != -1) {
		switch(c) {
			case 'c':
				printf("Got lcore_mask = %s\n", optarg);
				// atoi() doesn't work for hex representation
				lcore_mask = strtol(optarg, NULL, 16);
				break;
			default:
				red_printf("Master needs coremask. Exiting!\n");
				exit(-1);
		}
	}

	assert(lcore_mask != -1);
	red_printf("Master got lcore_mask: %d\n", lcore_mask);

	red_printf("Master: Creating worker-master shared queues\n");
	assert(WM_MAX_LCORE * sizeof(struct wm_queue) < M_2);
	wmq = (volatile struct wm_queue *) shm_alloc(WM_QUEUE_KEY, M_2);
	red_printf("Master: Creating worker-master queues done\n");

	sleep(10000);
}
