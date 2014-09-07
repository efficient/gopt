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

/**
 * wmq : the worker/master queue for all worker lcores
 */
void master_gpu(volatile struct wm_queue *wmq, 
	int num_workers, int *worker_lcores)
{
	int i;
	long long prev_head[WM_MAX_LCORE] = {0};
	int worker_index = 0;
	volatile struct wm_queue *lc_wmq;		// Work queue of one worker

	assert(worker_lcores != NULL);

	while(1) {
		// Grab the lcore_id of the next worker
		int w_lcore_id = worker_lcores[worker_index];
		lc_wmq = &wmq[w_lcore_id];

		// Snapshot this worker queue's head
		long long w_head = lc_wmq->head;

		if(w_head != prev_head[w_lcore_id]) {
			// Iterate over the new packets
			for(i = prev_head[w_lcore_id]; i < w_head; i ++) {
				int q_i = i & WM_QUEUE_CAP_;		// Offset in the queue
				int ip_addr = lc_wmq->ipv4_address[q_i];

				lc_wmq->ports[q_i] = 0 + (ip_addr & 3);
			}
			
			prev_head[w_lcore_id] = w_head;
			/*printf("Master updating tail for worker (lcore %d) to %lld\n", 
				w_lcore_id, w_head);*/
			lc_wmq->tail = w_head;
		}
		
		// Round-robin among the workers
		worker_index ++;
		if(worker_index == num_workers) {
			worker_index = 0;
		}
	}
}

int main(int argc, char **argv)
{
	int c, i;
	int lcore_mask = -1;
	volatile struct wm_queue *wmq;

	while ((c = getopt (argc, argv, "c:")) != -1) {
		switch(c) {
			case 'c':
				printf("\tGPU master: Got lcore_mask = %s\n", optarg);
				// atoi() doesn't work for hex representation
				lcore_mask = strtol(optarg, NULL, 16);
				break;
			default:
				red_printf("\tGPU master: I need coremask. Exiting!\n");
				exit(-1);
		}
	}

	assert(lcore_mask != -1);
	red_printf("\tGPU master: got lcore_mask: %d\n", lcore_mask);

	red_printf("\tGPU master: creating worker-master shm queues\n");
	assert(WM_MAX_LCORE * sizeof(struct wm_queue) < M_2);
	wmq = (volatile struct wm_queue *) shm_alloc(WM_QUEUE_KEY, M_2);

	for(i = 0; i < WM_MAX_LCORE; i ++) {
		uint64_t c1 = (uint64_t) (uintptr_t) &wmq[i].head;
		uint64_t c2 = (uint64_t) (uintptr_t) &wmq[i].tail;
		uint64_t c3 = (uint64_t) (uintptr_t) &wmq[i].sent;

		// Ensure that all counters are in separate cachelines
		assert((c1 % 64 == 0) && (c2 % 64 == 0) && (c3 % 64 == 0));
	}

	red_printf("\tGPU master: creating worker-master shm queues done\n");

	int num_workers = bitcount(lcore_mask);
	int *worker_lcores = get_active_bits(lcore_mask);
	
	// Launch the GPU code
	master_gpu(wmq, num_workers, worker_lcores);
	
}
