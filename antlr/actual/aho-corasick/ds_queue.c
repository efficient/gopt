#include<stdio.h>
#include<stdlib.h>
#include<assert.h>

#include "ds_queue.h"

void ds_queue_init(struct ds_queue *q)
{
	// ds_queue_printf("Initializing queue %p\n", q);
	q->head = NULL;
	q->tail = NULL;
	q->count = 0;
}

void ds_queue_add(struct ds_queue *q, int data)
{
	ds_queue_printf("ds_queue: Adding data %d to ds_queue %p\n", data, q);

	/* Create a new null-terminated node */
	struct ds_qnode *new_node = malloc(sizeof(struct ds_qnode));
	assert(new_node != NULL);

	new_node->data = data;
	new_node->next = NULL;

	/* If the queue is empty */
	if(q->head == NULL) {
		q->head = new_node;
		q->tail = new_node;
	} else {
		q->tail->next = new_node;
		q->tail = new_node;
	}

	q->count++;
}

int ds_queue_remove(struct ds_queue *q)
{
	ds_queue_printf("ds_queue: Removing from ds_queue %p\n", q);

	int data;
	assert(q->head != NULL);

	struct ds_qnode *old_head;
	old_head = q->head;
	data = old_head->data;

	q->head = q->head->next;
	q->count--;

	if(q->head == NULL) {
		assert(q->count == 0);
		q->tail = NULL;
	}

	free(old_head);

	return data;
}

inline int ds_queue_is_empty(const struct ds_queue *q)
{
	if(q->count == 0) {
		return 1;
	}
	return 0;
}

int ds_queue_size(struct ds_queue *q)
{
	return q->count;
}


void ds_queue_free(struct ds_queue *q)
{
	while(ds_queue_size(q) != 0) {
		ds_queue_remove(q);
	}
}

void ds_queue_print(struct ds_queue *q)
{
	assert(q != NULL);
	struct ds_qnode *t = q->head;
	while(t != NULL) {
		printf("%d ", t->data);
		t = t->next;
	}
	printf("\n");
}

