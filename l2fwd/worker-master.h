// Capacity of a queue between a PS-style worker thread and the master thread
#define WM_QUEUE_CAP 1024
#define WM_QUEUE_CAP_ 1023

// Maximum outstanding packets maintained by a worker for the master
#define WM_QUEUE_THRESH 128
#define WM_QUEUE_KEY 1

// Maximum worker lcores supported by the master
#define WM_MAX_LCORE 16

/**
 * A shared queue between a worker and a master.
 * The mbufs are actually pointers to rte_mbuf structs, but we use void 
 * here to avoid using DPDK's include files here.
 */
struct wm_queue
{
	void *mbufs[WM_QUEUE_CAP];			/** < Book-keeping by worker thread */
	int ipv4_address[WM_QUEUE_CAP];		/** < Input by worker thread */
	int ports[WM_QUEUE_CAP];			/** < Output by master thread */

	/** < All counters should be on separate cachelines */
	long long head;		/** < Number of packets in queue */
	long long pad_1[7];	/** < Master's counter gets different cacheline */

	long long tail;		/** < Number of packets processed by the master */
	long long pad_2[7];	/** < Each wm_queue should be cacheline aligned */

	long long sent;		/** < Number of queue packets TX-ed */
	long long pad_3[7];
};

