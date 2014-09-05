#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

#include "worker-master.h"
#include "sizes.h"

// Re-define RTE_MAX_LCORE because we don't want to include
// any DPDK header files here

#define WM_MAX_LCORE 64

volatile struct wm_queue *wmq;

/**
 * Create worker-master queues for all possible lcores.
 */
void create_wm_queues()
{	
	// The worker-master queues should fit inside one hugepage
	assert(WM_MAX_LCORE * sizeof(struct wm_queue) < M_2);

	int shm_flags = IPC_CREAT | 0666 | SHM_HUGETLB;
	int sid = shmget(WM_QUEUE_KEY, M_2, shm_flags);
	if(sid == -1) {
		fprintf(stderr, "shmget Error! Failed to shm_alloc\n");
		int doh = system("cat /sys/devices/system/node/*/meminfo | grep Huge");
		exit(doh);
	}

	wmq = (volatile struct wm_queue *) shmat(sid, 0, 0);
	assert(wmq != NULL);
	
	memset((char *) wmq, 0, M_2);
	
}

int main()
{
	printf("\033[31mMaster: Creating worker-master shared queues\033[0m\n");
	create_wm_queues();
	printf("\t\033[31mMaster: Creating worker-master queues done\033[0m\n");
	
	sleep(10000);
}
