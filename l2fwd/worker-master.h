// Capacity of a queue between a PS-style worker thread and the master thread
#define WM_QUEUE_CAP 1024
#define WM_QUEUE_CAP_ 1023

// Maximum outstanding packets maintained by a worker for the master
#define WM_QUEUE_THRESH 128
#define WM_QUEUE_KEY 1

/**
 * A shared queue between a worker and a master.
 * The mbufs are actually pointers to rte_mbuf structs, but we use void 
 * here to avoid using DPDK's include files here.
 */
struct wm_queue
{
	void *mbufs[WM_QUEUE_CAP];
	int ipv4_address[WM_QUEUE_CAP];
	long long head;
	long long tail;
};

