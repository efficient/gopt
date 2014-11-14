#define DS_QUEUE_DBG 0

#define ds_queue_printf(...) \
	do { \
		if(DS_QUEUE_DBG == 1) { \
			printf(__VA_ARGS__); \
		} \
	} while(0)

struct ds_qnode {
	int data;
	struct ds_qnode *next;
};

struct ds_queue {
	struct ds_qnode *head, *tail;
	int count;
};

void ds_queue_init(struct ds_queue *q);
void ds_queue_add(struct ds_queue *q, int data);
int ds_queue_remove(struct ds_queue *q);
int ds_queue_size(struct ds_queue *q);
void ds_queue_free(struct ds_queue *q);
inline int ds_queue_is_empty(const struct ds_queue *q);
void ds_queue_print(struct ds_queue *q);
