#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

// nvcc assumes that all header files are C++ files. Tell it
// that these are C header files
extern "C" {
#include "ipv4.h"
#include "worker-master.h"
#include "util.h"
}

__global__ void
ipv4Gpu(int *ips, uint8_t *ports, uint8_t *ipv4_cache, int num_ips)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_ips) {
		int ip_addr = ips[i];
		ports[i] = ipv4_cache[ip_addr & IPv4_CACHE_CAP_];
	}
}

/**
 * wmq : the worker/master queue for all lcores. Non-NULL iff the lcore
 * 		 is an active worker.
 */
void master_gpu(volatile struct wm_queue *wmq, 
	int *h_ips, int *d_ips,						/**< Kernel inputs */
	uint8_t *h_ports, uint8_t *d_ports,			/**< Kernel outputs */
	uint8_t *d_ipv4_cache,
	int num_workers, int *worker_lcores)
{
	assert(num_workers != 0);
	assert(worker_lcores != NULL);
	
	int i, err;

	/** < The h_ips buffer start index for an lcore during a kernel launch */
	int ips_lo[WM_MAX_LCORE] = {0};
	int ips_cur = 0;

	/** <  Value of the queue-head from an lcore during the last iteration*/
	long long prev_head[WM_MAX_LCORE] = {0}, new_head[WM_MAX_LCORE] = {0};
	
	int w_it, w_lid;		/**< A worker-iterator and the worker's lcore-id */
	volatile struct wm_queue *lc_wmq;	/**< Work queue of one worker */

	while(1) {
		/**< Copy all the IP-addresses supplied by workers into the 
		  * contiguous h_ips buffer */
		for(w_it = 0; w_it < num_workers; w_it ++) {
			w_lid = worker_lcores[w_it];		// Don't use w_it after this
			lc_wmq = &wmq[w_lid];
			
			// Start the GPU-buffer extent for this lcore
			ips_lo[w_lid] = ips_cur;

			// Snapshot this worker queue's head
			new_head[w_lid] = lc_wmq->head;

			// Iterate over the new packets
			for(i = prev_head[w_lid]; i < new_head[w_lid]; i ++) {
				int q_i = i & WM_QUEUE_CAP_;	// Offset in this wrkr's queue
				int ip_addr = lc_wmq->ipv4_address[q_i];

				h_ips[ips_cur] = ip_addr;			
				ips_cur ++;
			}
		}

		if(ips_cur == 0) {		// Number of IPs = ips_cur
			continue;
		}

		/**< Copy packets to device */
		err = cudaMemcpy(d_ips, h_ips, ips_cur * WM_INPUT_SIZE, 
			cudaMemcpyHostToDevice);
		CPE(err != cudaSuccess, "Failed to copy IPs from host to device\n");

		/**< Kernel launch */
		int threadsPerBlock = 256;
		int blocksPerGrid = (ips_cur + threadsPerBlock - 1) / threadsPerBlock;
	
		ipv4Gpu<<<blocksPerGrid, threadsPerBlock>>>(d_ips, 
			d_ports, d_ipv4_cache, ips_cur);
		err = cudaGetLastError();
		CPE(err != cudaSuccess, "Failed to launch ipv4Gpu kernel\n");

		err = cudaMemcpy(h_ports, d_ports, ips_cur * WM_OUTPUT_SIZE,
			cudaMemcpyDeviceToHost);
		CPE(err != cudaSuccess, "Failed to copy ports from device to host\n");

		/**< Copy the ports back to worker queues */
		for(w_it = 0; w_it < num_workers; w_it ++) {
			w_lid = worker_lcores[w_it];		// Don't use w_it after this

			lc_wmq = &wmq[w_lid];
			for(i = prev_head[w_lid]; i < new_head[w_lid]; i ++) {
				int q_i = i & WM_QUEUE_CAP_;	// Offset in this wrkr's queue
				int ips_i = ips_lo[w_lid] + (i - prev_head[w_lid]);
				lc_wmq->ports[q_i] = h_ports[ips_i];
			}

			prev_head[w_lid] = new_head[w_lid];
		
			/** < Update tail for this worker */
			lc_wmq->tail = new_head[w_lid];
		}

		ips_cur = 0;
	}
}

int main(int argc, char **argv)
{
	int c, i, err = cudaSuccess;
	int lcore_mask = -1;
	volatile struct wm_queue *wmq;

	/** < CUDA buffers */
	int *h_ips, *d_ips;
	uint8_t *h_ports, *d_ports;	
	uint8_t *h_ipv4_cache, *d_ipv4_cache;

	/**< Get the worker lcore mask */
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

	/**< Allocate hugepages for the shared queues */
	red_printf("\tGPU master: creating worker-master shm queues\n");
	int wm_queue_bytes = M_2;
	while(wm_queue_bytes < WM_MAX_LCORE * sizeof(struct wm_queue)) {
		wm_queue_bytes += M_2;
	}
	printf("\t\tTotal size of wm_queues = %d hugepages\n", 
		wm_queue_bytes / M_2);
	wmq = (volatile struct wm_queue *) shm_alloc(WM_QUEUE_KEY, wm_queue_bytes);

	for(i = 0; i < WM_MAX_LCORE; i ++) {
		uint64_t c1 = (uint64_t) (uintptr_t) &wmq[i].head;
		uint64_t c2 = (uint64_t) (uintptr_t) &wmq[i].tail;
		uint64_t c3 = (uint64_t) (uintptr_t) &wmq[i].sent;

		// Ensure that all counters are in separate cachelines
		assert((c1 % 64 == 0) && (c2 % 64 == 0) && (c3 % 64 == 0));
	}

	red_printf("\tGPU master: creating worker-master shm queues done\n");

	/** < Allocate buffers for IP addresses from all workers*/
	red_printf("\tGPU master: creating buffers for IP addresses\n");
	int ips_buf_size = WM_QUEUE_CAP * WM_MAX_LCORE * WM_INPUT_SIZE;
	err = cudaMallocHost((void **) &h_ips, ips_buf_size);
	err = cudaMalloc((void **) &d_ips, ips_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMalloc IP buffers\n");

	/** < Allocate buffers for ports from all workers */
	red_printf("\tGPU master: creating buffers for ports\n");
	int ports_buf_size = WM_QUEUE_CAP * WM_MAX_LCORE * WM_OUTPUT_SIZE;
	err = cudaMallocHost((void **) &h_ports, ports_buf_size);
	err = cudaMalloc((void **) &d_ports, ports_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMalloc port buffers\n");

	/** < Create the IPv4 cache and copy it over */
	red_printf("\tGPU master: creating IPv4 cache\n");
	ipv4_cache_init(&h_ipv4_cache, IPv4_PORT_MASK);

	int ipv4_buf_size = IPv4_CACHE_CAP * sizeof(uint8_t);
	err = cudaMalloc((void **) &d_ipv4_cache, ipv4_buf_size);
	cudaMemcpy(d_ipv4_cache, h_ipv4_cache, ipv4_buf_size, 
		cudaMemcpyHostToDevice);

	int num_workers = bitcount(lcore_mask);
	int *worker_lcores = get_active_bits(lcore_mask);
	
	/** < Launch the GPU master */
	red_printf("\tGPU master: launching GPU code\n");
	master_gpu(wmq, 
		h_ips, d_ips, 
		h_ports, d_ports, 
		d_ipv4_cache,
		num_workers, worker_lcores);
	
}
