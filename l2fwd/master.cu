#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

/**< NVCC assumes that all header files are C++ files. Tell it that these are
  *  C header files. */
extern "C" {
#include "city.h"
#include "ndn.h"
#include "worker-master.h"
#include "util.h"
}

#define MASTER_TEST_GPU 0	/**< Test NDN hash table impl and exit */

/**< CityHash64 CUDA functions */
static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static const uint64_t k1 = 0xb492b66fbe98f273ULL;
static const uint64_t k2 = 0x9ae16a3b2f90404fULL;
static const uint64_t k3 = 0xc949d7c7509e6557ULL;

__device__ uint64_t cu_Rotate(uint64 val, int shift) {
	return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
}

__device__ uint64_t cu_RotateByAtLeast1(uint64_t val, int shift) {
	return (val >> shift) | (val << (64 - shift));
}

__device__ uint64_t cu_ShiftMix(uint64_t val) {
	return val ^ (val >> 47);
}

__device__ uint64_t cu_Fetch32(const char *s)
{
	uint32_t result;
	memcpy(&result, s, 4);
	return (uint64_t) result;
}

__device__ uint64_t cu_Fetch64(const char *s)
{
	uint64_t result;
	memcpy(&result, s, 8);
	return result;
}

__device__ uint64_t cu_Hash128to64(const uint128 x) {
	const uint64_t kMul = 0x9ddfea08eb382d69ULL;
	uint64_t a = (Uint128Low64(x) ^ Uint128High64(x)) * kMul;
	a ^= (a >> 47);
	uint64_t b = (Uint128High64(x) ^ a) * kMul;

	b ^= (b >> 47);
	b *= kMul;
	return b;
}

__device__ uint64_t cu_HashLen16(uint64_t u, uint64_t v)
{
	uint128 result;
	result.first = u;
	result.second = v;
	return cu_Hash128to64(result);
}

__device__ uint64_t cu_HashLen17to32(const char *s, size_t len) {
	uint64_t a = cu_Fetch64(s) * k1;
	uint64_t b = cu_Fetch64(s + 8);
	uint64_t c = cu_Fetch64(s + len - 8) * k2;
	uint64_t d = cu_Fetch64(s + len - 16) * k0;
	return cu_HashLen16(cu_Rotate(a - b, 43) + cu_Rotate(c, 30) + d,
		a + cu_Rotate(b ^ k3, 20) - c + len);
}

__device__ uint64_t cu_HashLen0to16(const char *s, size_t len)
{
	if (len > 8) {
		uint64_t a = cu_Fetch64(s);
		uint64_t b = cu_Fetch64(s + len - 8);
		return cu_HashLen16(a, cu_RotateByAtLeast1(b + len, (int) len)) ^ b;
	}

	if (len >= 4) {
		uint64_t a = cu_Fetch32(s);
		return cu_HashLen16(len + (a << 3), cu_Fetch32(s + len - 4));
	}

	if (len > 0) {
		uint8_t a = (uint8_t) s[0];
		uint8_t b = (uint8_t) s[len >> 1];
		uint8_t c = (uint8_t) s[len - 1];
		uint32_t y = (uint32_t) (a) + ((uint32_t) (b) << 8);
		uint32_t z = (uint32_t) len + ((uint32_t) (c) << 2);
		return cu_ShiftMix(y * k2 ^ z * k3) * k2;
	}

	return k2;
}

__device__ uint64_t cu_CityHash64(char *s, int len)
{
	if(len <= 16) {
		return cu_HashLen0to16(s, len);
	} else {
		return cu_HashLen17to32(s, len);
	}
}

/**< Kernel to compute CityHash64 of strings with length <= 32. Only used for
  *  testing my CUDA impl of CityHash64. */
__global__ void
hashGpu(struct wm_trace *req, uint32_t *resp, int num_reqs)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_reqs) {
		char *ptr = (char *) req[i].bytes;
		int len = ((uint8_t) ptr[0] & 31) + 1;

		uint64_t hash = cu_CityHash64(ptr, len);

		resp[i] = (uint32_t) ((hash & 0xffffffff) ^ (hash >> 32));
	}
}

/**< Try to find a match for the 1st component of this name in the hash
  *  table. This function gets called when the clever matching trick of
  *  starting from 2nd component matches fails. */
__device__ int cu_lookup_one_component(char *name, struct ndn_bucket *ht)
{
	int c_i, i;	/**< URL char iterator and slot iterator */
	int bkt_num, bkt_1, bkt_2;

	for(c_i = 0; c_i < NDN_TRACE_LEN; c_i ++) {
		if(name[c_i] == '/') {
			break;
		}
	}

	/**< c_i is now at the boundary of the 1st component */
	uint64_t prefix_hash = cu_CityHash64(name, c_i + 1);
	uint16_t tag = prefix_hash >> 48;

	struct ndn_slot *slots;

	/**< name[0] -> name[c_i] is a prefix of length c_i + 1 */
	for(bkt_num = 1; bkt_num <= 2; bkt_num ++) {
		if(bkt_num == 1) {
			bkt_1 = prefix_hash & NDN_NUM_BKT_;
			slots = ht[bkt_1].slots;
		} else {
			bkt_2 = (bkt_1 ^ cu_CityHash64((char *) &tag, 2)) & NDN_NUM_BKT_;
			slots = ht[bkt_2].slots;
		}

		/**< Now, "slots" points to an ndn_bucket. Find a valid slot
		  *  that contains the same hash. */
		for(i = 0; i < NDN_NUM_SLOTS; i ++) {
			int8_t _dst_port = slots[i].dst_port;
			uint64_t _hash = slots[i].cityhash;

			if(_dst_port >= 0 && _hash == prefix_hash) {

				/**< As we're only matching this component, we're done! */
				return slots[i].dst_port;
			}
		}
	}

	/**< No match even for the 1st component? */
	return -1;
}

/**< Kernel to look up NDN names in an NDN hash table */
__global__ void
ndnGpu(struct wm_trace *req, int *resp, struct ndn_bucket *ht, int num_reqs)
{
	int t_i = blockDim.x * blockIdx.x + threadIdx.x;

	if(t_i < num_reqs) {
		
		char *trace = (char *) req[t_i].bytes;

		int fwd_port = -1;
		int c_i, i;	/**< URL char iterator and slot iterator */
		int bkt_num, bkt_1 = 0, bkt_2 = 0;

		int terminate = 0;			/**< Stop processing this URL? */
		int prefix_match_found = 0;	/**< Stop this hash-table lookup ? */

		/**< Completely ignore 1-component matches */		
		for(c_i = 0; c_i < NDN_TRACE_LEN; c_i ++) {
			if(trace[c_i] == '/') {
				break;
			}
		}
		c_i ++;

		for(; c_i < NDN_TRACE_LEN; c_i ++) {
			if(trace[c_i] != '/') {
				continue;
			}

			/**< c_i is now at the boundary of a component longer than the 1st */
			uint64_t prefix_hash = cu_CityHash64(trace, c_i + 1);
			uint16_t tag = prefix_hash >> 48;

			struct ndn_slot *slots;

			/**< trace[0] -> trace[c_i] is a prefix of length c_i + 1 */
			for(bkt_num = 1; bkt_num <= 2; bkt_num ++) {
				if(bkt_num == 1) {
					bkt_1 = prefix_hash & NDN_NUM_BKT_;
					slots = ht[bkt_1].slots;
				} else {
					bkt_2 = (bkt_1 ^ cu_CityHash64((char *) &tag, 2)) & 
						NDN_NUM_BKT_;
					slots = ht[bkt_2].slots;
				}

				/**< Now, "slots" points to an ndn_bucket. Find a valid slot
				  *  that contains the same hash. */
				for(i = 0; i < NDN_NUM_SLOTS; i ++) {
					int8_t _dst_port = slots[i].dst_port;
					uint64_t _hash = slots[i].cityhash;

					if(_dst_port >= 0 && _hash == prefix_hash) {

						/**< Record the dst port: this may get overwritten by
						  *  longer prefix matches later */
						fwd_port = slots[i].dst_port;

						if(slots[i].is_terminal == 1) {
							/**< A terminal FIB entry: we're done! */
							terminate = 1;
						}

						prefix_match_found = 1;
						break;
					}
				}

				/**< Stop the hash-table lookup for trace[0 ... c_i] */
				if(prefix_match_found == 1) {
					break;
				}
			}

			/**< Stop processing the trace if we found a terminal FIB entry */
			if(terminate == 1) {
				break;
			}
		}	/**< Loop over URL characters ends here */

		/**< We failed to match with prefixes that contain 2 or more
		  *  components. Try matching the 1st component of this trace now */
		if(fwd_port == -1) {
			fwd_port = cu_lookup_one_component(trace, ht);
		}

		resp[t_i] = fwd_port;
	}
}

/**< wmq: the worker/master queue for all lcores. Non-NULL iff the lcore is an
  *  active worker. */
void master_gpu(volatile struct wm_queue *wmq, cudaStream_t my_stream,
	struct wm_trace *h_reqs, struct wm_trace *d_reqs,	/**< Kernel inputs */
	int *h_resps, int *d_resps,	/**< Kernel outputs */
	struct ndn_bucket *d_ht,	/**< NDN hash table */
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
					(uint8_t *) lc_wmq->reqs[q_i].bytes, WM_REQ_SIZE);
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
	
		ndnGpu<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_reqs, 
			d_resps, d_ht, nb_req);
		err = cudaGetLastError();
		CPE(err != cudaSuccess, "Failed to launch ndnGpu kernel\n");

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

/**< Test the CUDA kernel by comparing outputs with NDN lib
  *  outputs. Enabled by setting MASTER_TEST_GPU = 1
  *  We need both host and device hash index for lookup comparison. */
void test_ndn_gpu(int nb_req,
	cudaStream_t my_stream, struct ndn_name *name_arr, int nb_names,
	struct wm_trace *h_reqs, struct wm_trace *d_reqs,	/**< Kernel inputs */
	int *h_resps, int *d_resps,	/**< Kernel outputs */
	struct ndn_bucket *h_ht, struct ndn_bucket *d_ht)
{
	int i, err;
	assert(name_arr != NULL && h_reqs != NULL && d_reqs != NULL &&
		h_resps != NULL && d_resps != NULL && d_ht != NULL);

	/**< Ensure that requests will fit in the allocated worker-master queues */
	assert(nb_req <= WM_MAX_LCORE * WM_QUEUE_CAP);

	/**< Create random probe traces from inserted names */
	for(i = 0; i < nb_req; i ++) {
		int name_i = rand() % nb_names;
		char *trace_ptr = (char *) &h_reqs[i];

		int name_len = strlen((char *) name_arr[name_i].bytes);

		/**< Copy name to trace. This will stay in range of the name because
		  *  NDN_MAX_NAME_LENGTH > NDN_TRACE_LEN */
		memcpy(trace_ptr, name_arr[name_i + i].bytes, NDN_TRACE_LEN);

		/**< Extend or truncate the trace to exactly 32 bytes */
		if(name_len < NDN_TRACE_LEN) {
			int c_i;
			for(c_i = name_len; c_i < NDN_TRACE_LEN; c_i ++) {
				trace_ptr[c_i] = 'a' + (rand() & 0xf);
			}
			trace_ptr[NDN_TRACE_LEN] = 0;
		} else {
			trace_ptr[NDN_TRACE_LEN] = 0;
		}
	}
	
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	/**< Copy requests to device */
	err = cudaMemcpyAsync(d_reqs, h_reqs, nb_req * WM_REQ_SIZE,
		cudaMemcpyHostToDevice, my_stream);
	CPE1(err != cudaSuccess, "Failed to copy requests h2d. nb_req = %d\n",
		nb_req);

	/**< Kernel launch */
	int threadsPerBlock = 256;
	int blocksPerGrid = (nb_req + threadsPerBlock - 1) / threadsPerBlock;

	ndnGpu<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_reqs,
		d_resps, d_ht, nb_req);
	err = cudaGetLastError();
	CPE(err != cudaSuccess, "Failed to launch ndnGpu kernel\n");

	/**< Copy responses from device */
	err = cudaMemcpyAsync(h_resps, d_resps, nb_req * WM_RESP_SIZE,
		cudaMemcpyDeviceToHost, my_stream);
	CPE(err != cudaSuccess, "Failed to copy responses d2h\n");

	/**< Synchronize all CUDA operations */
	cudaStreamSynchronize(my_stream);

	clock_gettime(CLOCK_REALTIME, &end);

	int nb_fails = 0;

	for(i = 0; i < nb_req; i ++) {
		/**< Casting from wm_trace to ndn_trace should be fine. */
		int exp_next_hop = ndn_lookup_gpu_only((struct ndn_trace *) &h_reqs[i],
			h_ht);
		if(exp_next_hop == -1) {
			nb_fails ++;
		}

		/**< Compare with kernel output */
		if(exp_next_hop != h_resps[i]) {
			printf("Probe %d failed! NDN: %d, CUDA: %d\n",
				i, exp_next_hop, h_resps[i]);
			exit(-1);
		}
	}

	double seconds = ((double) (end.tv_nsec - start.tv_nsec)) / 1000000000.0 +
		(end.tv_sec - start.tv_sec);
	printf("\t\tGPU NDN lookup test passed! "
		"GPU lookup rate = %.1f M/s\n", (nb_req / seconds) / 1000000);

}

/**< Test the CUDA implementation of CityHash32.
  *  Enabled by setting MASTER_TEST_GPU = 1 */
void test_hash(int nb_req,
	cudaStream_t my_stream,
	struct wm_trace *h_reqs, struct wm_trace *d_reqs,	/**< Kernel inputs */
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
		for(j = 0; j < WM_TRACE_LEN; j ++) {
			h_reqs[i].bytes[j] = (uint8_t) fastrand_ni(&seed);
		}
	}
	
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	/**< Copy requests to device */
	err = cudaMemcpyAsync(d_reqs, h_reqs, nb_req * WM_REQ_SIZE,
		cudaMemcpyHostToDevice, my_stream);
	CPE1(err != cudaSuccess, "Failed to copy requests h2d. nb_req = %d\n",
		nb_req);

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

		/**< The number of bytes of the string to use for hash computation is
		  *  b/w 1 and 32 and is computed using the 1st byte of the string. */
		int hash_len = (h_reqs[i].bytes[0] & 31) + 1;
		uint64_t exp_hash = CityHash64((char *) &h_reqs[i], hash_len);

		/**< To reuse the resp buffer created for finding destination ports,
		  *  the GPU hash impl returns 32 bit integers. */
		uint32_t exp_val =
			(uint32_t) ((exp_hash & 0xffffffff) ^ (exp_hash >> 32));

		/**< Compare with kernel output */
		if(exp_val != h_resps[i]) {
			printf("Probe %d failed! CityHash: %x, CUDA: %x\n",
				i, exp_val, h_resps[i]);
		exit(-1);
		}
	}

	double seconds = ((double) (end.tv_nsec - start.tv_nsec)) / 1000000000.0 +
		(end.tv_sec - start.tv_sec);
	printf("\t\tCityHash64 test passed! "
		"GPU CityHash64 rate = %.1f M/s\n", (nb_req / seconds) / 1000000);
}

int main(int argc, char **argv)
{
	/**< NDN data structures */
	struct ndn_bucket *h_ht, *d_ht;
	struct ndn_name *name_arr;
	int nb_names;

	/**< CUDA request and response buffers */
	struct wm_trace *h_reqs, *d_reqs;
	int *h_resps, *d_resps;	

	/**< Sanity check */
	assert(NDN_TRACE_LEN == WM_TRACE_LEN);

	int c, i, err = cudaSuccess;
	int lcore_mask = -1;
	cudaStream_t my_stream;
	volatile struct wm_queue *wmq;

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

	/**< Read NDN names - these are used to generated traces later */
	red_printf("Counting and reading NDN names..\n");
	nb_names = ndn_get_num_lines(NDN_NAME_FILE);
	name_arr = ndn_get_name_array(NDN_NAME_FILE);
	red_printf("\tRead %d NDN names.\n", nb_names);

	/**< Use the name file to populate the FIB */
	red_printf("Creating NDN hash index..\n");
	ndn_init(NDN_NAME_FILE, NDN_PORTMASK, &h_ht);
	red_printf("\tCreating NDN hash index done!\n");
	blue_printf("\tGPU master: alloc-ing hash index on device\n");

	/**< Copy the hash table (FIB) to the GPU */
	int ht_bytes = NDN_NUM_BKT * sizeof(struct ndn_bucket);
	err = cudaMalloc((void **) &d_ht, ht_bytes);
	CPE(err != cudaSuccess, "Failed to cudaMalloc ht\n");
	err = cudaMemcpy(d_ht, h_ht, ht_bytes, cudaMemcpyHostToDevice);
	CPE(err != cudaSuccess, "Failed to cudaMemcpy NDN hash table\n");

	int num_workers = bitcount(lcore_mask);
	int *worker_lcores = get_active_bits(lcore_mask);
	
	/**< Launch the GPU master */
#if MASTER_TEST_GPU == 1
	int nb_req;
	for(nb_req = 32; nb_req <= WM_MAX_LCORE * WM_QUEUE_CAP; nb_req *= 2) {
		blue_printf("GPU master: testing with %d requests\n", nb_req);

		printf("\tTesting CityHash64 impl\n");
		test_hash(nb_req, my_stream,
			h_reqs, d_reqs,
			(uint32_t *) h_resps, (uint32_t *) d_resps);

		printf("\tTesting NDN hash table impl\n");
		test_ndn_gpu(nb_req, my_stream,
			name_arr, nb_names,
			h_reqs, d_reqs,
			h_resps, d_resps,
			h_ht, d_ht);
	}
#else
	blue_printf("\tGPU master: launching GPU code\n");
	printf("\t\tUnused name_arr = %p\n", name_arr);
	master_gpu(wmq, my_stream,
		h_reqs, d_reqs, 
		h_resps, d_resps, 
		d_ht,
		num_workers, worker_lcores);
#endif
}
