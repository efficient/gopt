#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

/**< NVCC assumes that all header files are C++ files. Tell it that these are
  *  C header files. */
extern "C" {
#include "cuckoo.h"
#include "worker-master.h"
#include "util.h"
}

#define MASTER_TEST_GPU 1	/**< Test cuckoo hash table impl and exit */

uint32_t non_inline_fastrand_for_cuda(uint64_t *seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (uint32_t) (*seed >> 32);
}

/**< Functions for hashing from within a CUDA kernel */
static const uint32_t cu_c1 = 0xcc9e2d51;
static const uint32_t cu_c2 = 0x1b873593;

__device__ uint32_t cu_Rotate32(uint32_t val, int shift) 
{
	return shift == 0 ? val : ((val >> shift) | (val << (32 - shift)));
}

__device__ uint32_t cu_Mur(uint32_t a, uint32_t h) 
{
	a *= cu_c1;
	a = cu_Rotate32(a, 17);
	a *= cu_c2;
	h ^= a;
	h = cu_Rotate32(h, 19);
	return h * 5 + 0xe6546b64;
}

__device__ uint32_t cu_fmix(uint32_t h)
{
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
	return h;
}

__device__ uint32_t cu_Hash32Len0to4(char *s, int len) 
{
	uint32_t b = 0;
	uint32_t c = 9;
	int i;
	for(i = 0; i < len; i++) {
		b = b * cu_c1 + s[i];
		c ^= b;
	}
	return cu_fmix(cu_Mur(b, cu_Mur(len, c)));
}
/**< Hashing functions for CUDA kernels end here */

/**< Kernel to look up 32-bit MAC addresses in a cuckoo hash table */
__global__ void
cuckooGpu(uint32_t *req, int *resp, struct cuckoo_bucket *ht_index, int num_reqs)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_reqs) {
		int bkt_1, bkt_2, fwd_port = -1;

		uint32_t dst_mac = req[i];
		bkt_1 = cu_Hash32Len0to4((char *) &dst_mac, 4) & NUM_BKT_;

		for(int j = 0; j < 8; j ++) {
			uint32_t slot_mac = ht_index[bkt_1].slot[j].mac;
			int slot_port = ht_index[bkt_1].slot[j].port;

			if(slot_mac == dst_mac) {
				fwd_port = slot_port;
				break;
				/**< Don't break: avoid divergence. TODO: Maybe bad idea? */
			}
		}

		if(fwd_port == -1) {
			bkt_2 = cu_Hash32Len0to4((char *) &bkt_1, 4) & NUM_BKT_;

			for(int j = 0; j < 8; j ++) {
				uint32_t slot_mac = ht_index[bkt_2].slot[j].mac;
				int slot_port = ht_index[bkt_2].slot[j].port;

				if(slot_mac == dst_mac) {
					fwd_port = slot_port;
					break;
					/**<Don't break: avoid divergence. TODO: Maybe bad idea? */
				}
			}
		}

		resp[i] = fwd_port;
	}
}

/**< Kernel to compute hashes of 32-bit ints. Only used for testing my CUDA
  *  impl of CityHash32 */
__global__ void
hashGpu(uint32_t *req, uint32_t *resp, int num_reqs)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_reqs) {
		uint32_t dst_mac = req[i];
		resp[i] = cu_Hash32Len0to4((char *) &dst_mac, 4);
	}
}

/**< wmq: the worker/master queue for all lcores. Non-NULL iff the lcore is an
  *  active worker. */
void master_gpu(volatile struct wm_queue *wmq, cudaStream_t my_stream,
	uint32_t *h_reqs, uint32_t *d_reqs,	/**< Kernel inputs */
	int *h_resps, int *d_resps,	/**< Kernel outputs */
	struct cuckoo_bucket *d_ht_index,	/**< MAC lookup index */
	int num_workers, int *worker_lcores)
{
	assert(num_workers != 0);
	assert(worker_lcores != NULL);
	assert(num_workers * WM_QUEUE_THRESH < WM_QUEUE_CAP);
	
	int i, err;

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
			
			/**< Snapshot this worker queue's head. The entries in the queue up
			  *  to index (lc_wmq->head - 1) are definitely valid. The entry at
			  *  index lc_wmq->head also might be valid in some cases - we will
			  *  process it in the next iteration */
			new_head[w_lid] = lc_wmq->head;

			/**< Record the beginning of the GPU req. buffer for this lcore */
			req_lo[w_lid] = nb_req;

			/**< Add the new requests from this worker to the GPU req. buffer */
			for(i = prev_head[w_lid]; i < new_head[w_lid]; i ++) {
				int q_i = i & WM_QUEUE_CAP_;	/**< Queues are circular */
				uint32_t req = lc_wmq->reqs[q_i];

				h_reqs[nb_req] = req;
				nb_req ++;
			}
		}

		if(nb_req == 0) {	/**< No new packets from any worker? */
			continue;
		}

		/**< Copy requests to device */
		err = cudaMemcpyAsync(d_reqs, h_reqs, nb_req * sizeof(uint32_t), 
			cudaMemcpyHostToDevice, my_stream);
		CPE(err != cudaSuccess, "Failed to copy requests h2d\n");

		/**< Kernel launch */
		int threadsPerBlock = 256;
		int blocksPerGrid = (nb_req + threadsPerBlock - 1) / threadsPerBlock;
	
		cuckooGpu<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_reqs, 
			d_resps, d_ht_index, nb_req);
		err = cudaGetLastError();
		CPE(err != cudaSuccess, "Failed to launch cuckooGpu kernel\n");

		/**< Copy responses from device */
		err = cudaMemcpyAsync(h_resps, d_resps, nb_req * sizeof(int),
			cudaMemcpyDeviceToHost, my_stream);
		CPE(err != cudaSuccess, "Failed to copy responses d2h\n");

		/**< Synchronize all CUDA operations */
		cudaStreamSynchronize(my_stream);
		
		/**< Copy the responses back to worker queues */
		for(w_i = 0; w_i < num_workers; w_i ++) {
			w_lid = worker_lcores[w_i];		/**< Don't use w_i after this */
			lc_wmq = &wmq[w_lid];

			for(i = prev_head[w_lid]; i < new_head[w_lid]; i ++) {
				/**< Offset in this workers' queue and the GPU req. buffer */
				int q_i = i & WM_QUEUE_CAP_;				
				int req_i = req_lo[w_lid] + (i - prev_head[w_lid]);
				lc_wmq->resps[q_i] = h_resps[req_i];
			}

			prev_head[w_lid] = new_head[w_lid];
		
			/**< Update tail for this worker */
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

/**< Test the CUDA kernel by comparing cuckoo outputs with cuckoo lib
  *  outputs. Enabled by setting MASTER_TEST_GPU = 1
  *  We need both host and device hash index for lookup comparison. */
void test_cuckoo_32_gpu(int nb_req,
	cudaStream_t my_stream, uint32_t *mac_addrs,
	uint32_t *h_reqs, uint32_t *d_reqs,	/**< Kernel inputs */
	int *h_resps, int *d_resps,	/**< Kernel outputs */
	struct cuckoo_bucket *d_ht_index, struct cuckoo_bucket *h_ht_index)
{
	int i, err;
	assert(mac_addrs != NULL && h_reqs != NULL && d_reqs != NULL &&
		h_resps != NULL && d_resps != NULL && d_ht_index != NULL);

	/**< Ensure that requests will fit in the allocated worker-master queues */
	assert(nb_req <= WM_MAX_LCORE * WM_QUEUE_CAP);

	/**< Choose random probe addresses from inserted MACs */
	for(i = 0; i < nb_req; i ++) {
		int prefix_arr_i = rand() % NUM_MAC_;
		h_reqs[i] = mac_addrs[prefix_arr_i];
	}
	
	/**< Copy requests to device */
	err = cudaMemcpyAsync(d_reqs, h_reqs, nb_req * sizeof(uint32_t),
		cudaMemcpyHostToDevice, my_stream);
	CPE1(err != cudaSuccess, "Failed to copy requests h2d. nb_req = %d\n",
		nb_req);

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	/**< Kernel launch */
	int threadsPerBlock = 256;
	int blocksPerGrid = (nb_req + threadsPerBlock - 1) / threadsPerBlock;

	cuckooGpu<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_reqs, 
		d_resps, d_ht_index, nb_req);
	err = cudaGetLastError();
	CPE(err != cudaSuccess, "Failed to launch ipv6Gpu kernel\n");

	/**< Copy responses from device */
	err = cudaMemcpyAsync(h_resps, d_resps, nb_req * sizeof(int),
		cudaMemcpyDeviceToHost, my_stream);
	CPE(err != cudaSuccess, "Failed to copy responses d2h\n");

	/**< Synchronize all CUDA operations */
	cudaStreamSynchronize(my_stream);

	clock_gettime(CLOCK_REALTIME, &end);

	for(i = 0; i < nb_req; i ++) {
		int exp_next_hop = cuckoo_lookup(h_reqs[i], h_ht_index);

		/**< Compare with kernel output */
		if(exp_next_hop != h_resps[i]) {
			printf("Probe %d failed! cuckoo: %d, CUDA: %d\n",
				i, exp_next_hop, h_resps[i]);
			exit(-1);
		}
	}

	double seconds = ((double) (end.tv_nsec - start.tv_nsec)) / 1000000000.0 +
		(end.tv_sec - start.tv_sec);
	printf("\t\tGPU hash table rate = %.1f M/s\n", (nb_req / seconds) / 1000000);
}

/**< Test the CUDA implementation of CityHash32.
  *  Enabled by setting MASTER_TEST_GPU = 1 */
void test_hash(int nb_req,
	cudaStream_t my_stream,
	uint32_t *h_reqs, uint32_t *d_reqs,	/**< Kernel inputs */
	uint32_t *h_resps, uint32_t *d_resps)	/**< Kernel outputs */
{
	int i, err;
	assert( h_reqs != NULL && d_reqs != NULL &&
		h_resps != NULL && d_resps != NULL);

	/**< Ensure that requests will fit in the allocated worker-master queues */
	assert(nb_req <= WM_MAX_LCORE * WM_QUEUE_CAP);

	/**< Create random uint32_t for testing */
	uint64_t seed = 0xdeadbeef;
	for(i = 0; i < nb_req; i ++) {
		h_reqs[i] = non_inline_fastrand_for_cuda(&seed);
	}
	
	/**< Copy requests to device */
	err = cudaMemcpyAsync(d_reqs, h_reqs, nb_req * sizeof(uint32_t),
		cudaMemcpyHostToDevice, my_stream);
	CPE1(err != cudaSuccess, "Failed to copy requests h2d. nb_req = %d\n",
		nb_req);

	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	/**< Kernel launch */
	int threadsPerBlock = 256;
	int blocksPerGrid = (nb_req + threadsPerBlock - 1) / threadsPerBlock;

	hashGpu<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_reqs, 
		d_resps, nb_req);
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
		int exp_hash = CityHash32((char *) &h_reqs[i], 4);

		/**< Compare with kernel output */
		if(exp_hash != h_resps[i]) {
			printf("Probe %d failed! CityHash: %d, CUDA: %d\n",
				i, exp_hash, h_resps[i]);
			exit(-1);
		}
	}

	double seconds = ((double) (end.tv_nsec - start.tv_nsec)) / 1000000000.0 +
		(end.tv_sec - start.tv_sec);
	printf("\t\tGPU CityHash32 rate = %.1f M/s\n", (nb_req / seconds) / 1000000);
}

int main(int argc, char **argv)
{
	int c, i, err = cudaSuccess;
	int lcore_mask = -1, nb_req;
	cudaStream_t my_stream;
	volatile struct wm_queue *wmq;

	/**< CUDA buffers */
	uint32_t *h_reqs, *d_reqs;
	int *h_resps, *d_resps;	

	struct cuckoo_bucket *h_ht_index, *d_ht_index;
	uint32_t *mac_addrs;

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
	wmq = (volatile struct wm_queue *) shm_alloc(WM_QUEUE_KEY, wm_queue_bytes);

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
	int reqs_buf_size = WM_QUEUE_CAP * WM_MAX_LCORE * sizeof(uint32_t);
	err = cudaMallocHost((void **) &h_reqs, reqs_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMallocHost req buffer\n");
	err = cudaMalloc((void **) &d_reqs, reqs_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMalloc req buffer\n");

	/**< Allocate buffers for responses for all workers */
	blue_printf("\tGPU master: creating buffers for responses\n");
	int resps_buf_size = WM_QUEUE_CAP * WM_MAX_LCORE * sizeof(int);
	err = cudaMallocHost((void **) &h_resps, resps_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMallocHost resp buffers\n");
	err = cudaMalloc((void **) &d_resps, resps_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMalloc resp buffers\n");

	/**< Create the cuckoo hash-index and copy it over */
	blue_printf("\tGPU master: creating cuckoo hash index\n");

	cuckoo_init(&mac_addrs, &h_ht_index, CUCKOO_PORT_MASK);

	int ht_index_bytes = NUM_BKT * sizeof(struct cuckoo_bucket);
	
	blue_printf("\tGPU master: alloc-ing hash index on device\n");
	err = cudaMalloc((void **) &d_ht_index, ht_index_bytes);
	CPE(err != cudaSuccess, "Failed to cudaMalloc ht_index\n");
	cudaMemcpy(d_ht_index, h_ht_index, ht_index_bytes, 
		cudaMemcpyHostToDevice);

	int num_workers = bitcount(lcore_mask);
	int *worker_lcores = get_active_bits(lcore_mask);
	
	/**< Launch the GPU master */
#if MASTER_TEST_GPU == 1
	for(nb_req = 32; nb_req <= WM_MAX_LCORE * WM_QUEUE_CAP; nb_req *= 2) {
		blue_printf("GPU master: testing with %d requests\n", nb_req);

		printf("\tTesting CityHash32 impl\n");
		test_hash(nb_req,
			my_stream,
			h_reqs, d_reqs,
			(uint32_t *) h_resps, (uint32_t *) d_resps);

		printf("\tTesting cuckoo hash table impl\n");
		test_cuckoo_32_gpu(nb_req,
			my_stream, mac_addrs,
			h_reqs, d_reqs,
			h_resps, d_resps,
			d_ht_index, h_ht_index);
	}
#else
	blue_printf("\tGPU master: launching GPU code\n");
	master_gpu(wmq, my_stream,
		h_reqs, d_reqs, 
		h_resps, d_resps, 
		d_ht_index,
		num_workers, worker_lcores);
#endif
}
