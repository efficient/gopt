/**< Capacity of a queue between a PacketShader-style worker-master */
#define WM_QUEUE_CAP 16384
#define WM_QUEUE_CAP_ 16383

/**< Maximum outstanding packets maintained by a worker for the master */
#define WM_QUEUE_THRESH 4096
#define WM_QUEUE_KEY 1

/**< Maximum worker lcores supported by the master */
#define WM_MAX_LCORE 16

struct wm_ipv6_addr {
	uint8_t bytes[16];
};

/**
 * A shared circular queue between a worker and a master.
 * The mbufs are actually pointers to rte_mbuf structs, but we use void 
 * here to avoid using DPDK's include files.
 */
struct wm_queue
{
	void *mbufs[WM_QUEUE_CAP];	/**< Book keeping by worker thread */
	struct wm_ipv6_addr reqs[WM_QUEUE_CAP];	/**< Input by worker thread/host */
	uint16_t resps[WM_QUEUE_CAP];	/**< Output by master thread/device */

	/**< All counters should be on separate cachelines */
	long long head;		/**< Total number of packets ever queued by worker */
	long long pad_1[7];	/**< Master's counter gets different cacheline */

	long long tail;		/**< Total number of packets ever procd by master */
	long long pad_2[7];	/**< Each wm_queue should be cacheline aligned */

	long long sent;		/**< Number of queue packets TX-ed */
	long long pad_3[7];	/**< Each wm_queue should be cacheline aligned */
};

