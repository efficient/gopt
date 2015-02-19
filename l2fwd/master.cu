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

__device__ uint32_t cu_Fetch32(uint8_t *s)
{
	uint32_t byte_3 = s[0];
	uint32_t byte_2 = s[1];
	uint32_t byte_1 = s[2];
	uint32_t byte_0 = s[3];

	return (byte_0 << 24) + (byte_1 << 16) + (byte_2 << 8) + byte_3;
}

__device__ uint32_t cu_Hash32Len5to12(char *s, int len)
{
	uint32_t a = len, b = len * 5, c = 9, d = b;

	a += cu_Fetch32((uint8_t *) s);
	b += cu_Fetch32((uint8_t *) (s + len - 4));
	c += cu_Fetch32((uint8_t *) (s + ((len >> 1) & 4)));
	
  	return cu_fmix(cu_Mur(c, cu_Mur(b, cu_Mur(a, d))));
}

__device__ ULL cu_get_mac(uint8_t *mac_ptr)
{
	ULL ret = 0;
	ret = mac_ptr[0] + 
		((ULL) mac_ptr[1] << 8) + 
		((ULL) mac_ptr[2] << 16) +
		((ULL) mac_ptr[3] << 24) +
		((ULL) mac_ptr[4] << 32) +
		((ULL) mac_ptr[5] << 40);

	return ret;
}

/**< Kernel to look up MAC addresses in a cuckoo hash table */
__global__ void
cuckooGpu(struct wm_ether_addr *req,
	int *resp, struct cuckoo_bucket *ht_index, int num_reqs)
{
	int t_i = blockDim.x * blockIdx.x + threadIdx.x;

	if(t_i < num_reqs) {
		
		int bkt_1, bkt_2, fwd_port = -1;
		int i;

		uint8_t *dst_mac_ptr = req[t_i].addr_bytes;
		ULL dst_mac = cu_get_mac(dst_mac_ptr);

		/**< Compute the 1st bucket using the full mac address */
		bkt_1 = cu_Hash32Len5to12((char *) dst_mac_ptr, 6) & NUM_BKT_;

		for(i = 0; i < 8; i ++) {
			if(SLOT_TO_MAC(ht_index[bkt_1].slot[i]) == dst_mac) {
				fwd_port = SLOT_TO_PORT(ht_index[bkt_1].slot[i]);
				break;
			}
		}

		/**< 2nd bucket is computed using the 1st bucket */
		if(fwd_port == -1) {
			bkt_2 = cu_Hash32Len0to4((char *) &bkt_1, 4) & NUM_BKT_;

			for(i = 0; i < 8; i ++) {
				if(SLOT_TO_MAC(ht_index[bkt_2].slot[i]) == dst_mac) {
					fwd_port = SLOT_TO_PORT(ht_index[bkt_2].slot[i]);
					break;
				}
			}
		}

		resp[t_i] = fwd_port;
	}
}

/**< Kernel to compute hashes of 4 byte and 6 byte strings. Only used for
  *  testing my CUDA impl of CityHash32 */
__global__ void
hashGpu(struct wm_ether_addr *req, uint32_t *resp, int num_reqs)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_reqs) {
		uint8_t *dst_mac_ptr = req[i].addr_bytes;
		uint32_t hash_6 = cu_Hash32Len5to12((char *) dst_mac_ptr, 6);
		uint32_t hash_4 = cu_Hash32Len0to4((char *) dst_mac_ptr, 4);
		resp[i] = hash_6 ^ hash_4;
	}
}

/**< wmq: the worker/master queue for all lcores. Non-NULL iff the lcore is an
  *  active worker. */
void master_gpu(volatile struct wm_queue *wmq, cudaStream_t my_stream,
	struct wm_ether_addr *h_reqs, struct wm_ether_addr *d_reqs,	/**< Kernel inputs */
	int *h_resps, int *d_resps,	/**< Kernel outputs */
	struct cuckoo_bucket *d_ht_index,	/**< MAC lookup index */
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
				memcpy((uint8_t *) &h_reqs[nb_req],
					(uint8_t *) &lc_wmq->reqs[q_i], WM_REQ_SIZE);

				nb_req ++;
			}
		}

		if(nb_req == 0) {	/**< No new packets from any worker? */
			continue;
		}

		/**< Copy requests to device */
		err = cudaMemcpyAsync(d_reqs, h_reqs, nb_req * WM_REQ_SIZE,
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
		err = cudaMemcpyAsync(h_resps, d_resps, nb_req * WM_RESP_SIZE,
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

/**< Test the CUDA kernel by comparing cuckoo outputs with cuckoo lib
  *  outputs. Enabled by setting MASTER_TEST_GPU = 1
  *  We need both host and device hash index for lookup comparison. */
void test_cuckoo_48_gpu(int nb_req,
	cudaStream_t my_stream, ULL *mac_addrs,
	struct wm_ether_addr *h_reqs, struct wm_ether_addr *d_reqs,	/**< Kernel inputs */
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
		set_mac_ni(h_reqs[i].addr_bytes, mac_addrs[prefix_arr_i]);
	}
	
	/**< Copy requests to device */
	err = cudaMemcpyAsync(d_reqs, h_reqs, nb_req * WM_REQ_SIZE,
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
	err = cudaMemcpyAsync(h_resps, d_resps, nb_req * WM_RESP_SIZE,
		cudaMemcpyDeviceToHost, my_stream);
	CPE(err != cudaSuccess, "Failed to copy responses d2h\n");

	/**< Synchronize all CUDA operations */
	cudaStreamSynchronize(my_stream);

	clock_gettime(CLOCK_REALTIME, &end);

	int nb_fails = 0;

	for(i = 0; i < nb_req; i ++) {
		int exp_next_hop = cuckoo_lookup(h_reqs[i].addr_bytes, h_ht_index);
		if(exp_next_hop == -1) {
			nb_fails ++;
		}

		/**< Compare with kernel output */
		if(exp_next_hop != h_resps[i]) {
			printf("Probe %d failed! cuckoo: %d, CUDA: %d\n",
				i, exp_next_hop, h_resps[i]);
			exit(-1);
		}
	}

	double seconds = ((double) (end.tv_nsec - start.tv_nsec)) / 1000000000.0 +
		(end.tv_sec - start.tv_sec);
	printf("\t\tGPU hash table rate = %.1f M/s, fail percentage = %.3f\n",
		(nb_req / seconds) / 1000000, (float) nb_fails / nb_req);
}

/**< Test the CUDA implementation of CityHash32.
  *  Enabled by setting MASTER_TEST_GPU = 1 */
void test_hash(int nb_req,
	cudaStream_t my_stream,
	struct wm_ether_addr *h_reqs, struct wm_ether_addr *d_reqs,	/**< Kernel inputs */
	uint32_t *h_resps, uint32_t *d_resps)	/**< Kernel outputs */
{
	int i, j, err;
	assert(h_reqs != NULL && d_reqs != NULL &&
		h_resps != NULL && d_resps != NULL);

	/**< Ensure that requests will fit in the allocated worker-master queues */
	assert(nb_req <= WM_MAX_LCORE * WM_QUEUE_CAP);

	/**< Create random MAC addresses for testing */
	uint64_t seed = 0xdeadbeef;
	for(i = 0; i < nb_req; i ++) {
		for(j = 0; j < 6; j ++) {
			h_reqs[i].addr_bytes[j] = (uint8_t) fastrand_ni(&seed);
		}
	}
	
	/**< Copy requests to device */
	err = cudaMemcpyAsync(d_reqs, h_reqs, nb_req * WM_REQ_SIZE,
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
	err = cudaMemcpyAsync(h_resps, d_resps, nb_req * WM_RESP_SIZE,
		cudaMemcpyDeviceToHost, my_stream);
	CPE(err != cudaSuccess, "Failed to copy responses d2h\n");

	/**< Synchronize all CUDA operations */
	cudaStreamSynchronize(my_stream);

	clock_gettime(CLOCK_REALTIME, &end);

	for(i = 0; i < nb_req; i ++) {
		int exp_hash = CityHash32((char *) &h_reqs[i], 6) ^
			CityHash32((char *) &h_reqs[i], 4);

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
	int lcore_mask = -1;
	cudaStream_t my_stream;
	volatile struct wm_queue *wmq;

	/**< CUDA buffers */
	struct wm_ether_addr *h_reqs, *d_reqs;
	int *h_resps, *d_resps;	

	struct cuckoo_bucket *h_ht_index, *d_ht_index;
	ULL *mac_addrs;

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
	int max_reqs = WM_QUEUE_CAP * WM_MAX_LCORE;

	blue_printf("\tGPU master: creating buffers for requests\n");
	int reqs_buf_size = max_reqs * WM_REQ_SIZE;
	err = cudaMallocHost((void **) &h_reqs, reqs_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMallocHost req buffer\n");
	err = cudaMalloc((void **) &d_reqs, reqs_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMalloc req buffer\n");

	/**< Allocate buffers for responses for all workers */
	blue_printf("\tGPU master: creating buffers for responses\n");
	int resps_buf_size = max_reqs * WM_RESP_SIZE;
	err = cudaMallocHost((void **) &h_resps, resps_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMallocHost resp buffers\n");
	err = cudaMalloc((void **) &d_resps, resps_buf_size);
	CPE(err != cudaSuccess, "Failed to cudaMalloc resp buffers\n");

	/**< Create the cuckoo hash-index and copy it over */
	blue_printf("\tGPU master: creating cuckoo hash index\n");
	cuckoo_init(&mac_addrs, &h_ht_index, CUCKOO_PORT_MASK);

	blue_printf("\tGPU master: alloc-ing hash index on device\n");
	int ht_index_bytes = NUM_BKT * sizeof(struct cuckoo_bucket);
	err = cudaMalloc((void **) &d_ht_index, ht_index_bytes);
	CPE(err != cudaSuccess, "Failed to cudaMalloc ht_index\n");
	cudaMemcpy(d_ht_index, h_ht_index, ht_index_bytes, 
		cudaMemcpyHostToDevice);

	int num_workers = bitcount(lcore_mask);
	int *worker_lcores = get_active_bits(lcore_mask);
	
	/**< Launch the GPU master */
#if MASTER_TEST_GPU == 1
	int nb_req;
	for(nb_req = 32; nb_req <= WM_MAX_LCORE * WM_QUEUE_CAP; nb_req *= 2) {
		blue_printf("GPU master: testing with %d requests\n", nb_req);

		printf("\tTesting CityHash32 impl\n");
		test_hash(nb_req,
			my_stream,
			h_reqs, d_reqs,
			(uint32_t *) h_resps, (uint32_t *) d_resps); /**< Hash -> uint32 */

		printf("\tTesting cuckoo hash table impl\n");
		test_cuckoo_48_gpu(nb_req,
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
