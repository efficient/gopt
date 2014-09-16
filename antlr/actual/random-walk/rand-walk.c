#include "rand-walk.h"

void rand_walk_init(struct node **nodes)
{
	int i, j;

	/** < Allocate the nodes */
	printf("\tInitializing nodes for random walk. Size = %lu bytes\n", 
		NUM_NODES * sizeof(struct node));

	int sid = shmget(RAND_WALK_KEY, NUM_NODES * sizeof(struct node), 
		IPC_CREAT | 0666 | SHM_HUGETLB);

	if(sid < 0) {
		printf("\tCould not create nodes for random walk\n");
		exit(-1);
	}

	*nodes = shmat(sid, 0, 0);
	memset((char *) *nodes, 0, NUM_NODES * sizeof(struct node));

	/** < Initialize nodes with random pointers */
	printf("\tInitializing all nodes\n");

	for(i = 0; i < NUM_NODES; i++) {
		(*nodes)[i].id = i;
		for(j = 0; j < 7; j ++) {
			/** < Compute a random neighbor */
			int nbh_id = rand() & NUM_NODES_;
			(*nodes)[i].neighbors[j] = (void *) &((*nodes)[nbh_id]);
		}	
	}

}


void red_printf(const char *format, ...)
{	
	#define RED_LIM 1000
	va_list args;
	int i;

	char buf1[RED_LIM], buf2[RED_LIM];
	memset(buf1, 0, RED_LIM);
	memset(buf2, 0, RED_LIM);

    va_start(args, format);

	// Marshal the stuff to print in a buffer
	vsnprintf(buf1, RED_LIM, format, args);

	// Probably a bad check for buffer overflow
	for(i = RED_LIM - 1; i >= RED_LIM - 50; i --) {
		assert(buf1[i] == 0);
	}

	// Add markers for red color and reset color
	snprintf(buf2, 1000, "\033[31m%s\033[0m", buf1);

	// Probably another bad check for buffer overflow
	for(i = RED_LIM - 1; i >= RED_LIM - 50; i --) {
		assert(buf2[i] == 0);
	}

	printf("%s", buf2);

    va_end(args);
}

