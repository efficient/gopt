#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

/**< NVCC assumes that all header files are C++ files. Tell it that these are
  *  C header files. */
extern "C" {
#include "ipv6.h"
#include "worker-master.h"
#include "util.h"
}

#define MASTER_TEST_GPU 0

__global__ void
ipv6Gpu(struct ipv6_addr *req, uint32_t *resp, 
	uint32_t *tbl24, uint32_t *tbl8,
	int num_reqs)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	int status;
	uint8_t first_byte;
	uint32_t tbl24_index, tbl8_index, tbl_entry;
	uint8_t *ip;

	if (i < num_reqs) {
		ip = req[i].bytes;

		first_byte = LOOKUP_FIRST_BYTE;
		tbl24_index = (ip[0] << BYTES2_SIZE) | (ip[1] << BYTE_SIZE) | ip[2];

		tbl_entry = tbl24[tbl24_index];
		
		do {
			if ((tbl_entry & RTE_LPM6_VALID_EXT_ENTRY_BITMASK) ==
				RTE_LPM6_VALID_EXT_ENTRY_BITMASK) {
				
				tbl8_index = ip[first_byte - 1] +
					((tbl_entry & RTE_LPM6_TBL8_BITMASK) *
					RTE_LPM6_TBL8_GROUP_NUM_ENTRIES);

				tbl_entry = tbl8[tbl8_index];
				first_byte ++;
				status = 1;
			} else {
				resp[i] = tbl_entry;
				status = 0;
			}
		} while (status == 1);
	}
}

/**< wmq: the worker/master queue for all lcores. Non-NULL iff the lcore is an
  *  active worker. */
void master_gpu(volatile struct wm_queue *wmq, cudaStream_t my_stream,
	struct ipv6_addr *h_reqs, struct ipv6_addr *d_reqs,	/**< Kernel inputs */
	uint32_t *h_resps, uint32_t *d_resps,	/**< Outputs, tbl_entry ~ 4B */
	uint32_t *d_tbl24, uint32_t *d_tbl8,	/**< IPv6 lookup tables */
	int num_workers, int *worker_lcores)
{
	assert(num_workers != 0);
	assert(worker_lcores != NULL);
	assert(num_workers <= WM_MAX_LCORE);
	
	/**< A circular queue iterator. This needs to be 64 bit to go from the old
	  *  queue head to the new head. */
	long long cq_i;
	int err;

	/**< Variables for batch-size and latency averaging measurements */
	int msr_iter = 0;			/**< Number of kernel launches */
	long long msr_tot_req = 0;	/**< Total packet serviced by the master */
	struct timespec msr_start, msr_end;
	double msr_tot_us = 0;		/**< Total microseconds over all iterations */

	/**< The GPU-buffer (h_reqs) start index for a worker's packets during a
	  *  kernel launch. */
	int req_lo[WM_MAX_LCORE] = {0};

	/**< Number of requests that we'll send to the GPU = nb_req. We don't need
	  *  to worry about nb_req overflowing the capacity of h_reqs because it
	  *  fits all WM_MAX_LCORE. */
	int nb_req = 0;

	/**< Value of the queue-head from an lcore during the last iteration*/
	long long prev_head[WM_MAX_LCORE] = {0}, new_head[WM_MAX_LCORE] = {0};
	
	int w_i, w_lid;		/**< A worker-iterator and the worker's lcore-id */
	volatile struct wm_queue *lc_wmq;	/**< Work queue of one worker */

	clock_gettime(CLOCK_REALTIME, &msr_start);

	while(1) {

		/**< Copy all the requests supplied by workers into the contiguous 
		  *  h_reqs buffer. */
		for(w_i = 0; w_i < num_workers; w_i ++) {
			w_lid = worker_lcores[w_i];		/**< Don't use w_i after this */
			lc_wmq = &wmq[w_lid];
			
			/**< Snapshot this worker queue's head. lc_wmq->head is the number
			  *  entries queued by this worker, so the entries in the queue up
			  *  to index (lc_wmq->head - 1) are definitely valid. */
			new_head[w_lid] = lc_wmq->head;

			/**< Record the beginning of the GPU req. buffer for this lcore */
			req_lo[w_lid] = nb_req;

			/**< Add the new requests from this worker to the GPU req. buffer */
			for(cq_i = prev_head[w_lid]; cq_i < new_head[w_lid]; cq_i ++) {
				int q_i = cq_i & WM_QUEUE_CAP_;	/**< Actual queue offset */

				ipv6_mov16((uint8_t *) &h_reqs[nb_req],
					(uint8_t *) lc_wmq->reqs[q_i].bytes);
				nb_req ++;
			}
		}

		if(nb_req == 0) {	/**< No new packets from any worker? */
			continue;
		}

		/**< Copy requests to device */
		err = cudaMemcpyAsync(d_reqs, h_reqs, nb_req * sizeof(struct ipv6_addr), 
			cudaMemcpyHostToDevice, my_stream);
		CPE(err != cudaSuccess, "Failed to copy requests h2d\n");

		/**< Kernel launch */
		int threadsPerBlock = 256;
		int blocksPerGrid = (nb_req + threadsPerBlock - 1) / threadsPerBlock;
	
		ipv6Gpu<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_reqs, 
			d_resps, d_tbl24, d_tbl8, nb_req);
		err = cudaGetLastError();
		CPE(err != cudaSuccess, "Failed to launch ipv6Gpu kernel\n");

		/**< Copy responses from device */
		err = cudaMemcpyAsync(h_resps, d_resps, nb_req * sizeof(uint32_t),
			cudaMemcpyDeviceToHost, my_stream);
		CPE(err != cudaSuccess, "Failed to copy responses d2h\n");

		/**< Synchronize all CUDA operations */
		cudaStreamSynchronize(my_stream);
		
		/**< Copy the responses back to worker queues */
		for(w_i = 0; w_i < num_workers; w_i ++) {
			w_lid = worker_lcores[w_i];		/**< Don't use w_i after this */
			lc_wmq = &wmq[w_lid];

			for(cq_i = prev_head[w_lid]; cq_i < new_head[w_lid]; cq_i ++) {
				/**< Offset in this workers' queue and the GPU req. buffer */
				int q_i = cq_i & WM_QUEUE_CAP_;				
				int req_i = req_lo[w_lid] + (cq_i - prev_head[w_lid]);
				lc_wmq->resps[q_i] = h_resps[req_i];
			}

			prev_head[w_lid] = new_head[w_lid];
		
			/**< Update tail for this worker: the master has processed packets
			  *  up to index (new_head[w_lid] - 1) for this worker, so total
			  *  number of processed packets is new_head[w_lid]. */
			lc_wmq->tail = new_head[w_lid];
		}

		/**< Do some GPU-specific measurements */
		msr_iter ++;
		msr_tot_req += nb_req;

		if(msr_iter == 100000) {
			clock_gettime(CLOCK_REALTIME, &msr_end);
			msr_tot_us = (msr_end.tv_sec - msr_start.tv_sec) * 1000000 +
				(msr_end.tv_nsec - msr_start.tv_nsec) / 1000;

			blue_printf("\tGPU master: average batch size = %lld\n"
				"\t\tAverage time for GPU communication = %f us\n",
				msr_tot_req / msr_iter, msr_tot_us / msr_iter);

			msr_iter = 0;
			msr_tot_req = 0;

			/**< Start the next measurement */
			clock_gettime(CLOCK_REALTIME, &msr_start);
		}

		nb_req = 0;
	}
}

/**< Test the CUDA kernel by comparing prefix matching outputs with DPDK's
  *  outputs. Enabled by setting MASTER_TEST_GPU = 1 */
void test_ipv6_gpu(struct rte_lpm6 *lpm, int nb_req,
	cudaStream_t my_stream,
	struct ipv6_prefix *prefix_arr,
	struct ipv6_addr *h_reqs, struct ipv6_addr *d_reqs,	/**< Kernel inputs */
	uint32_t *h_resps, uint32_t *d_resps,	/**< Outputs, tbl_entry ~ 4B */
	uint32_t *d_tbl24, uint32_t *d_tbl8)	/**< IPv6 lookup tables */
{
			
	int i, err;
	assert(prefix_arr != NULL);

	/**< Ensure that requests will fit in the allocated worker-master queues */
	assert(nb_req <= WM_MAX_LCORE * WM_QUEUE_CAP);

	/**< Choose random probe addresses from inserted prefixes */
	for(i = 0; i < nb_req; i ++) {
		int prefix_arr_i = rand() % IPV6_NUM_RAND_PREFIXES;
		memcpy(h_reqs[i].bytes, prefix_arr[prefix_arr_i].bytes, IPV6_ADDR_LEN);
	}
	
	/**< Copy requests to device */
	err = cudaMemcpyAsync(d_reqs, h_reqs, nb_req * IPV6_ADDR_LEN, 
		cudaMemcpyHostToDevice, my_stream);
	CPE1(err != cudaSuccess, "Failed to copy requests h2d. nb_req = %d\n",
		nb_req);

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	/**< Kernel launch */
	int threadsPerBlock = 256;
	int blocksPerGrid = (nb_req + threadsPerBlock - 1) / threadsPerBlock;

	ipv6Gpu<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_reqs, 
		d_resps, d_tbl24, d_tbl8, nb_req);
	err = cudaGetLastError();
	CPE(err != cudaSuccess, "Failed to launch ipv6Gpu kernel\n");

	/**< Copy responses from device */
	err = cudaMemcpyAsync(h_resps, d_resps, nb_req * sizeof(uint32_t),
		cudaMemcpyDeviceToHost, my_stream);
	CPE(err != cudaSuccess, "Failed to copy responses d2h\n");

	/**< Synchronize all CUDA operations */
	cudaStreamSynchronize(my_stream);

	clock_gettime(CLOCK_REALTIME, &end);

	for(i = 0; i < nb_req; i ++) {
		uint8_t exp_next_hop;
		rte_lpm6_lookup(lpm, h_reqs[i].bytes, &exp_next_hop);

		/**< Compare DPDK's output with kernel output */
		if((uint8_t) exp_next_hop != (uint8_t) h_resps[i]) {
			printf("Probe %d failed! DPDK: %d, CUDA: %d\n",
				i, exp_next_hop, (uint8_t) h_resps[i]);
			exit(-1);
		}
	}

	red_printf("GPU IPv6 tests passed (total = %d)\n", nb_req);

	double seconds = ((double) (end.tv_nsec - start.tv_nsec)) / 1000000000.0 +
		(end.tv_sec - start.tv_sec);
	red_printf("GPU IPv6 rate = %.1f M/s\n", (nb_req / seconds) / 1000000);
}

int main(int argc, char **argv)
{
	int c, i, err = cudaSuccess;
	int lcore_mask = -1;
	cudaStream_t my_stream;
	volatile struct wm_queue *wmq;

	/**< CUDA buffers */
	struct ipv6_addr *h_reqs, *d_reqs;
	uint32_t *h_resps, *d_resps;	
	uint32_t *d_tbl24, *d_tbl8;	/**< No need for host pinned memory */

	struct rte_lpm6 *lpm;
	struct ipv6_prefix *prefix_arr;

	/**< Get the worker lcore mask */
	while ((c = getopt (argc, argv, "c:")) != -1) {
		switch(c) {
			case 'c':
				/**< atoi() doesn't work for hex representation */
				lcore_mask = strtol(optarg, NULL, 16);
				break;
			default:
				blue_printf("\tGPU master: I need coremask. Exiting!\n");
				exit(-1);
		}
	}

	assert(lcore_mask != -1);
	blue_printf("\tGPU master: got lcore_mask: %x\n", lcore_mask);

	/**< Create a CUDA stream */
	err = cudaStreamCreate(&my_stream);
	CPE(err != cudaSuccess, "Failed to create cudaStream\n");

	/**< Allocate hugepages for the shared queues */
	blue_printf("\tGPU master: creating worker-master shm queues\n");
	int wm_queue_bytes = M_2;
	while(wm_queue_bytes < WM_MAX_LCORE * sizeof(struct wm_queue)) {
		wm_queue_bytes += M_2;
	}
	printf("\t\tTotal size of wm_queues = %d hugepages\n", 
		wm_queue_bytes / M_2);
	wmq = (volatile struct wm_queue *) hrd_malloc_socket(WM_QUEUE_KEY,
		wm_queue_bytes, 0);

	/**< Ensure that queue counters are in separate cachelines */
	for(i = 0; i < WM_MAX_LCORE; i ++) {
		uint64_t c1 = (uint64_t) (uintptr_t) &wmq[i].head;
		uint64_t c2 = (uint64_t) (uintptr_t) &wmq[i].tail;
		uint64_t c3 = (uint64_t) (uintptr_t) &wmq[i].sent;

		assert((c1 % 64 == 0) && (c2 % 64 == 0) && (c3 % 64 == 0));
	}

	blue_printf("\tGPU master: creating worker-master shm queues done\n");

	/**< Allocate buffers for requests from all workers*/
	blue_printf("\tGPU master: creating buffers for requests\n");
	int reqs_buf_size = WM_QUEUE_CAP * WM_MAX_LCORE * sizeof(struct ipv6_addr);
	err = cudaMallocHost((void **) &h_reqs, reqs_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMallocHost req buffer\n");
	err = cudaMalloc((void **) &d_reqs, reqs_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMalloc req buffer\n");

	/**< Allocate buffers for responses for all workers */
	blue_printf("\tGPU master: creating buffers for responses\n");
	int resps_buf_size = WM_QUEUE_CAP * WM_MAX_LCORE * sizeof(uint32_t);
	err = cudaMallocHost((void **) &h_resps, resps_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMallocHost resp buffers\n");
	err = cudaMalloc((void **) &d_resps, resps_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMalloc resp buffers\n");

	/**< Create the IPv6 LPM structure and copy it over */
	blue_printf("\tGPU master: creating rte_lpm lookup table\n");
	lpm = ipv6_init(IPV6_XIA_R2_PORT_MASK, &prefix_arr, 1);

	/**< rte_lpm6_tbl_entry is 4 bytes */
	int entry_sz = sizeof(struct rte_lpm6_tbl_entry);
	int tbl24_bytes = RTE_LPM6_TBL24_NUM_ENTRIES * entry_sz;
	int tbl8_bytes = (IPV6_NUM_TBL8 * RTE_LPM6_TBL8_GROUP_NUM_ENTRIES) *
		entry_sz;
	
	/**< Alloc and copy tbl24 and tbl8 arrays to GPU memory */
	blue_printf("\tGPU master: alloc tbl24 (size = %d bytes) on device\n",
		tbl24_bytes);
	err = cudaMalloc((void **) &d_tbl24, tbl24_bytes);
	CPE(err != cudaSuccess, "Failed to cudaMalloc tbl24\n");
	cudaMemcpy(d_tbl24, lpm->tbl24, tbl24_bytes, cudaMemcpyHostToDevice);

	blue_printf("\tGPU master: alloc tbl8 (size = %d bytes) on device\n",
		tbl8_bytes);
	err = cudaMalloc((void **) &d_tbl8, tbl8_bytes);
	CPE(err != cudaSuccess, "Failed to cudaMalloc tbl8\n");
	cudaMemcpy(d_tbl8, lpm->tbl8, tbl8_bytes, cudaMemcpyHostToDevice);

	int num_workers = bitcount(lcore_mask);
	int *worker_lcores = get_active_bits(lcore_mask);
	
	/**< Launch the GPU master */
#if MASTER_TEST_GPU == 1
	blue_printf("\tGPU master: launching GPU test code\n");
	int nb_req = 32;
	for(; nb_req <= WM_MAX_LCORE * WM_QUEUE_CAP; nb_req *= 2) {
		test_ipv6_gpu(lpm, nb_req,
			my_stream,
			prefix_arr,
			h_reqs, d_reqs, 
			h_resps, d_resps, 
			d_tbl24, d_tbl8);
	}

#else
	blue_printf("\tGPU master: launching GPU real code\n");
	master_gpu(wmq, my_stream,
		h_reqs, d_reqs, 
		h_resps, d_resps, 
		d_tbl24, d_tbl8,
		num_workers, worker_lcores);
#endif
}
