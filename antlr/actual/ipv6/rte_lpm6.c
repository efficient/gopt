/*-
 *   BSD LICENSE
 * 
 *   Copyright(c) 2010-2013 Intel Corporation. All rights reserved.
 *   All rights reserved.
 * 
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 * 
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <numaif.h>

#include "rte_lpm6.h"
#include "fpp.h"

#define RTE_LPM6_TBL24_NUM_ENTRIES        (1 << 24)
#define RTE_LPM6_TBL8_GROUP_NUM_ENTRIES         256
#define RTE_LPM6_TBL8_MAX_NUM_GROUPS      (1 << 21)

#define RTE_LPM6_VALID_EXT_ENTRY_BITMASK 0xA0000000
#define RTE_LPM6_LOOKUP_SUCCESS          0x20000000
#define RTE_LPM6_TBL8_BITMASK            0x001FFFFF

#define ADD_FIRST_BYTE                            3
#define LOOKUP_FIRST_BYTE                         4
#define BYTE_SIZE                                 8
#define BYTES2_SIZE                              16

#define lpm6_tbl8_gindex next_hop

/** Flags for setting an entry as valid/invalid. */
enum valid_flag {
	INVALID = 0,
	VALID
};

/** Tbl entry structure. It is the same for both tbl24 and tbl8 */
struct rte_lpm6_tbl_entry {
	uint32_t next_hop:	21;  /**< Next hop / next table to be checked. */
	uint32_t depth	:8;      /**< Rule depth. */
	
	/* Flags. */
	uint32_t valid     :1;   /**< Validation flag. */
	uint32_t valid_group :1; /**< Group validation flag. */
	uint32_t ext_entry :1;   /**< External entry. */
};

/** Rules tbl entry structure. */
struct rte_lpm6_rule {
	uint8_t ip[RTE_LPM6_IPV6_ADDR_SIZE]; /**< Rule IP address. */
	uint8_t next_hop; /**< Rule next hop. */
	uint8_t depth; /**< Rule depth. */
};

/** LPM6 structure. */
struct rte_lpm6 {
	/* LPM metadata. */
	char name[RTE_LPM6_NAMESIZE];    /**< Name of the lpm. */
	uint32_t max_rules;              /**< Max number of rules. */
	uint32_t used_rules;             /**< Used rules so far. */
	uint32_t number_tbl8s;           /**< Number of tbl8s to allocate. */
	uint32_t next_tbl8;              /**< Next tbl8 to be used. */

	/* LPM Tables. */
	struct rte_lpm6_rule *rules_tbl; /**< LPM rules. */
	struct rte_lpm6_tbl_entry tbl24[RTE_LPM6_TBL24_NUM_ENTRIES]
			__rte_cache_aligned; /**< LPM tbl24 table. */
	struct rte_lpm6_tbl_entry tbl8[0]
			__rte_cache_aligned; /**< LPM tbl8 table. */
};

/**< Allocate size bytes in hugepages on this socket */
void *hrd_malloc_socket(int shm_key, int size, int socket_id)
{
	printf("rte_lpm6: Allocating %d MB (hugepg) on socket %d. SHM key = %d\n",
		size / (1024 * 1024), socket_id, shm_key);
	int shmid = shmget(shm_key, size, IPC_CREAT | 0666 | SHM_HUGETLB);
	assert(shmid >= 0);
	void *buf = shmat(shmid, 0, 0);
	assert(buf != NULL);

	/**< Bind the buffer to this socket */
	const unsigned long nodemask = (1 << socket_id);
	mbind(buf, size, MPOL_BIND, &nodemask, 32, 0);

	return buf;
}

/*
 * Takes an array of uint8_t (IPv6 address) and masks it using the depth.
 * It leaves untouched one bit per unit in the depth variable
 * and set the rest to 0.
 */
static inline void
mask_ip(uint8_t *ip, uint8_t depth)
{
        int16_t part_depth, mask;
        int i;
		
		part_depth = depth;

		for (i = 0; i < RTE_LPM6_IPV6_ADDR_SIZE; i++) {
			if (part_depth < BYTE_SIZE && part_depth >= 0) {
				mask = (uint16_t)(~(UINT8_MAX >> part_depth));
				ip[i] = (uint8_t)(ip[i] & mask);
			} else if (part_depth < 0) {
				ip[i] = 0;
			}
			part_depth -= BYTE_SIZE;
		}
}

/*
 * Allocates memory for LPM object
 */
struct rte_lpm6 *
rte_lpm6_create(int socket_id, const struct rte_lpm6_config *config)
{
	struct rte_lpm6 *lpm = NULL;
	uint64_t mem_size, rules_size;

	assert(sizeof(struct rte_lpm6_tbl_entry) == sizeof(uint32_t));

	/* Check user arguments. */
	assert(socket_id >= 0 && config != NULL && config->max_rules > 0 &&
		config->number_tbl8s <= RTE_LPM6_TBL8_MAX_NUM_GROUPS);

	/* Determine the amount of memory to allocate. */
	mem_size = sizeof(*lpm) + (sizeof(lpm->tbl8[0]) *
			RTE_LPM6_TBL8_GROUP_NUM_ENTRIES * config->number_tbl8s);
	rules_size = sizeof(struct rte_lpm6_rule) * config->max_rules;

	/* Allocate memory to store the LPM data structures. Zero out counters. */
	lpm = (struct rte_lpm6 *) hrd_malloc_socket(RTE_LPM6_SHM_KEY,
			mem_size, socket_id);
	assert(lpm != NULL);
	memset(lpm, 0, mem_size);
			
	/**< Not accessed on datapath */
	lpm->rules_tbl = (struct rte_lpm6_rule *) malloc(rules_size);
	assert(lpm->rules_tbl != NULL);
			
	/* Save user arguments. */
	lpm->max_rules = config->max_rules;
	lpm->number_tbl8s = config->number_tbl8s;

	return lpm;
}

/*
 * Checks if a rule already exists in the rules table and updates
 * the nexthop if so. Otherwise it adds a new rule if enough space is available.
 */
static inline int32_t
rule_add(struct rte_lpm6 *lpm, uint8_t *ip, uint8_t next_hop, uint8_t depth)
{
	uint32_t rule_index;
		
	/* Scan through rule list to see if rule already exists. */
	for (rule_index = 0; rule_index < lpm->used_rules; rule_index++) {

		/* If rule already exists update its next_hop and return. */
		if ((memcmp (lpm->rules_tbl[rule_index].ip, ip,
				RTE_LPM6_IPV6_ADDR_SIZE) == 0) &&
				lpm->rules_tbl[rule_index].depth == depth) {
			lpm->rules_tbl[rule_index].next_hop = next_hop;

			return rule_index;
		}
	}

	/*
	 * If rule does not exist check if there is space to add a new rule to
	 * this rule group. If there is no space return error.
	 */
	if (lpm->used_rules == lpm->max_rules) {
		return -ENOSPC;
	}

	/* If there is space for the new rule add it. */
	memcpy(lpm->rules_tbl[rule_index].ip, ip, RTE_LPM6_IPV6_ADDR_SIZE);
	lpm->rules_tbl[rule_index].next_hop = next_hop;
	lpm->rules_tbl[rule_index].depth = depth;

	/* Increment the used rules counter for this rule group. */
	lpm->used_rules++;
	
	return rule_index;
}

/*
 * Function that expands a rule across the data structure when a less-generic
 * one has been added before. It assures that every possible combination of bits
 * in the IP address returns a match.
 */
static void
expand_rule(struct rte_lpm6 *lpm, uint32_t tbl8_gindex, uint8_t depth,
		uint8_t next_hop)
{
	uint32_t tbl8_group_end, tbl8_gindex_next, j;

	tbl8_group_end = tbl8_gindex + RTE_LPM6_TBL8_GROUP_NUM_ENTRIES;

	struct rte_lpm6_tbl_entry new_tbl8_entry = {
		.valid = VALID,
		.valid_group = VALID,
		.depth = depth,
		.next_hop = next_hop,
		.ext_entry = 0,
	};

	for (j = tbl8_gindex; j < tbl8_group_end; j++) {
		if (!lpm->tbl8[j].valid || (lpm->tbl8[j].ext_entry == 0
				&& lpm->tbl8[j].depth <= depth)) {

			lpm->tbl8[j] = new_tbl8_entry;

		} else if (lpm->tbl8[j].ext_entry == 1) {

			tbl8_gindex_next = lpm->tbl8[j].lpm6_tbl8_gindex
					* RTE_LPM6_TBL8_GROUP_NUM_ENTRIES;
			expand_rule(lpm, tbl8_gindex_next, depth, next_hop);
		}
	}
}

/*
 * Partially adds a new route to the data structure (tbl24+tbl8s).
 * It returns 0 on success, a negative number on failure, or 1 if
 * the process needs to be continued by calling the function again.
 */
static inline int
add_step(struct rte_lpm6 *lpm, struct rte_lpm6_tbl_entry *tbl,
		struct rte_lpm6_tbl_entry **tbl_next, uint8_t *ip, uint8_t bytes,
		uint8_t first_byte, uint8_t depth, uint8_t next_hop)
{
	uint32_t tbl_index, tbl_range, tbl8_group_start, tbl8_group_end, i;
	int32_t tbl8_gindex;
	int8_t bitshift;
	uint8_t bits_covered;
	
	/*
	 * Calculate index to the table based on the number and position
	 * of the bytes being inspected in this step.
	 */
	tbl_index = 0;
	for (i = first_byte; i < (uint32_t)(first_byte + bytes); i++) {
		bitshift = (int8_t)((bytes - i)*BYTE_SIZE);
		
		if (bitshift < 0) bitshift = 0;
		tbl_index = tbl_index | ip[i-1] << bitshift;
	}

	/* Number of bits covered in this step */
	bits_covered = (uint8_t)((bytes+first_byte-1)*BYTE_SIZE);
	
	/*
	 * If depth if smaller than this number (ie this is the last step) 
	 * expand the rule across the relevant positions in the table.
	 */
	if (depth <= bits_covered) {
		tbl_range = 1 << (bits_covered - depth);

		for (i = tbl_index; i < (tbl_index + tbl_range); i++) {
			if (!tbl[i].valid || (tbl[i].ext_entry == 0 &&
					tbl[i].depth <= depth)) {

				struct rte_lpm6_tbl_entry new_tbl_entry = {
					.next_hop = next_hop,
					.depth = depth,
					.valid = VALID,
					.valid_group = VALID,
					.ext_entry = 0,
				};

				tbl[i] = new_tbl_entry;

			} else if (tbl[i].ext_entry == 1) {
			
				/*
				 * If tbl entry is valid and extended calculate the index
				 * into next tbl8 and expand the rule across the data structure.
				 */
				tbl8_gindex = tbl[i].lpm6_tbl8_gindex * 
						RTE_LPM6_TBL8_GROUP_NUM_ENTRIES;
				expand_rule(lpm, tbl8_gindex, depth, next_hop);
			}
		}    
		
		return 0;
	}
	/*
	 * If this is not the last step just fill one position
	 * and calculate the index to the next table.
	 */
	else {
		/* If it's invalid a new tbl8 is needed */
		if (!tbl[tbl_index].valid) {
			if (lpm->next_tbl8 < lpm->number_tbl8s)
				tbl8_gindex = (lpm->next_tbl8)++;
			else
				return -ENOSPC;
			
			struct rte_lpm6_tbl_entry new_tbl_entry = {
				.lpm6_tbl8_gindex = tbl8_gindex,
				.depth = 0,
				.valid = VALID,
				.valid_group = VALID,
				.ext_entry = 1,
			};

			tbl[tbl_index] = new_tbl_entry;
		}
		/*
		 * If it's valid but not extended the rule that was stored *
		 * here needs to be moved to the next table.
		 */
		else if (tbl[tbl_index].ext_entry == 0) {
			/* Search for free tbl8 group. */
			if (lpm->next_tbl8 < lpm->number_tbl8s)
				tbl8_gindex = (lpm->next_tbl8)++;
			else
				return -ENOSPC;

			tbl8_group_start = tbl8_gindex *
					RTE_LPM6_TBL8_GROUP_NUM_ENTRIES;
			tbl8_group_end = tbl8_group_start +
					RTE_LPM6_TBL8_GROUP_NUM_ENTRIES;

			/* Populate new tbl8 with tbl value. */
			for (i = tbl8_group_start; i < tbl8_group_end; i++) {
				lpm->tbl8[i].valid = VALID;
				lpm->tbl8[i].depth = tbl[tbl_index].depth;
				lpm->tbl8[i].next_hop = tbl[tbl_index].next_hop;
				lpm->tbl8[i].ext_entry = 0;
			}

			/*
			 * Update tbl entry to point to new tbl8 entry. Note: The
			 * ext_flag and tbl8_index need to be updated simultaneously,
			 * so assign whole structure in one go.
			 */
			struct rte_lpm6_tbl_entry new_tbl_entry = {
				.lpm6_tbl8_gindex = tbl8_gindex,
				.depth = 0,
				.valid = VALID,
				.valid_group = VALID,
				.ext_entry = 1,
			};

			tbl[tbl_index] = new_tbl_entry;
		}
		
		*tbl_next = &(lpm->tbl8[tbl[tbl_index].lpm6_tbl8_gindex *
				RTE_LPM6_TBL8_GROUP_NUM_ENTRIES]);
	}
	
	return 1;
}

/*
 * Add a route
 */
int
rte_lpm6_add(struct rte_lpm6 *lpm, uint8_t *ip, uint8_t depth,
		uint8_t next_hop)
{
	struct rte_lpm6_tbl_entry *tbl;
	struct rte_lpm6_tbl_entry *tbl_next;
	int32_t rule_index;
	int status;
	uint8_t masked_ip[RTE_LPM6_IPV6_ADDR_SIZE];
	int i;
	
	/* Check user arguments. */
	if ((lpm == NULL) || (depth < 1) || (depth > RTE_LPM6_MAX_DEPTH)) {
		printf("rte_lpm6: Invalid arguments to rte_lpm6_add\n");
		assert(0);
	}
		
	/* Copy the IP and mask it to avoid modifying user's input data. */
	memcpy(masked_ip, ip, RTE_LPM6_IPV6_ADDR_SIZE);
	mask_ip(masked_ip, depth);

	/* Add the rule to the rule table. */
	rule_index = rule_add(lpm, masked_ip, next_hop, depth);

	/* If there is no space available for new rule return error. */
	if (rule_index < 0) {
		printf("rte_lpm6: No space for new rule!\n");
		assert(0);
	}

	/* Inspect the first three bytes through tbl24 on the first step. */
	tbl = lpm->tbl24;
	status = add_step (lpm, tbl, &tbl_next, masked_ip, ADD_FIRST_BYTE, 1,
			depth, next_hop);
	if (status < 0) {
		rte_lpm6_delete(lpm, masked_ip, depth);
		
		return status;
	}

	/* 
	 * Inspect one by one the rest of the bytes until
	 * the process is completed.
	 */
	for (i = ADD_FIRST_BYTE; i < RTE_LPM6_IPV6_ADDR_SIZE && status == 1; i++) {
		tbl = tbl_next;
		status = add_step (lpm, tbl, &tbl_next, masked_ip, 1, (uint8_t)(i+1),
				depth, next_hop);
		if (status < 0) {
			rte_lpm6_delete(lpm, masked_ip, depth);
			
			return status;
		}
	}
	
	if(rand() % 100 == 0) {
		printf("rte_lpm6: next_tbl8 = %d\n", lpm->next_tbl8);
	}

	return status;
}

/*
 * Takes a pointer to a table entry and inspect one level.
 * The function returns 0 on lookup success, ENOENT if no match was found
 * or 1 if the process needs to be continued by calling the function again.
 */
static inline int
lookup_step(const struct rte_lpm6 *lpm, const struct rte_lpm6_tbl_entry *tbl,
		const struct rte_lpm6_tbl_entry **tbl_next, uint8_t *ip,
		uint8_t first_byte, uint8_t *next_hop)
{
	uint32_t tbl8_index, tbl_entry;
	
	/* Take the integer value from the pointer. */
	tbl_entry = *(const uint32_t *) tbl;
	
	/* If it is valid and extended we calculate the new pointer to return. */
	if ((tbl_entry & RTE_LPM6_VALID_EXT_ENTRY_BITMASK) ==
			RTE_LPM6_VALID_EXT_ENTRY_BITMASK) {

		tbl8_index = ip[first_byte - 1] +
				((tbl_entry & RTE_LPM6_TBL8_BITMASK) *
				RTE_LPM6_TBL8_GROUP_NUM_ENTRIES);

		*tbl_next = &lpm->tbl8[tbl8_index];

		return 1;
	} else {
		/* If not extended then we can have a match. */
		*next_hop = (uint8_t)tbl_entry;
		return (tbl_entry & RTE_LPM6_LOOKUP_SUCCESS) ? 0 : -ENOENT;
	}
}

/*
 * Looks up an IP
 */
int
rte_lpm6_lookup(const struct rte_lpm6 *lpm, uint8_t *ip, uint8_t *next_hop)
{
	const struct rte_lpm6_tbl_entry *tbl;
	const struct rte_lpm6_tbl_entry *tbl_next;
	int status;
	uint8_t first_byte;
	uint32_t tbl24_index;
	
	/* DEBUG: Check user input arguments. */
	if ((lpm == NULL) || (ip == NULL) || (next_hop == NULL)) {
		return -EINVAL;
	}
	
	first_byte = LOOKUP_FIRST_BYTE;
	tbl24_index = (ip[0] << BYTES2_SIZE) | (ip[1] << BYTE_SIZE) | ip[2];

	/* Calculate pointer to the first entry to be inspected */
	tbl = &lpm->tbl24[tbl24_index];
	
	do {
		/* Continue inspecting following levels until success or failure */		
		status = lookup_step(lpm, tbl, &tbl_next, ip, first_byte++, next_hop);
		tbl = tbl_next;
	} while (status == 1);
	
	return status;
}

/*
 * Looks up a group of IP addresses
 */
int
rte_lpm6_lookup_bulk_func(const struct rte_lpm6 *lpm,
		uint8_t ips[][RTE_LPM6_IPV6_ADDR_SIZE],
		int16_t * next_hops, unsigned n)
{
	unsigned i;
	const struct rte_lpm6_tbl_entry *tbl;
	const struct rte_lpm6_tbl_entry *tbl_next;
	uint32_t tbl24_index;
	uint8_t first_byte, next_hop;
	int status;

	/* DEBUG: Check user input arguments. */
	if ((lpm == NULL) || (ips == NULL) || (next_hops == NULL)) {
		return -EINVAL;
	}

	for (i = 0; i < n; i++) {
		first_byte = LOOKUP_FIRST_BYTE;
		tbl24_index = (ips[i][0] << BYTES2_SIZE) |
				(ips[i][1] << BYTE_SIZE) | ips[i][2];

		/* Calculate pointer to the first entry to be inspected */
		tbl = &lpm->tbl24[tbl24_index];
		
		do {
			/* Continue inspecting following levels until success or failure */		
			status = lookup_step(lpm, tbl, &tbl_next, ips[i], first_byte++,
					&next_hop);
			tbl = tbl_next;
		} while (status == 1);
		
		if (status < 0)
			next_hops[i] = -1;
		else
			next_hops[i] = next_hop;
	}
	
	return 0;
}

/*
 * Looks up a group of IP addresses
 */
int batch_index = 0;

void rte_lpm6_lookup_nogoto(const struct rte_lpm6 *lpm,
		uint8_t ips[][RTE_LPM6_IPV6_ADDR_SIZE],
		int16_t *next_hops, unsigned n)
{
	foreach(batch_index, n) {
		const struct rte_lpm6_tbl_entry *tbl;
		const struct rte_lpm6_tbl_entry *tbl_next;
		uint32_t tbl24_index;
		uint8_t first_byte, next_hop;
		int status;

		first_byte = LOOKUP_FIRST_BYTE;
		tbl24_index = (ips[batch_index][0] << BYTES2_SIZE) |
				(ips[batch_index][1] << BYTE_SIZE) | ips[batch_index][2];
		FPP_EXPENSIVE(&lpm->tbl24[tbl24_index]);

		/* Calculate pointer to the first entry to be inspected */
		tbl = &lpm->tbl24[tbl24_index];
		
		do {
			/* Continue inspecting following levels until success or failure */		
			status = lookup_step(lpm, tbl, &tbl_next, ips[batch_index], first_byte++,
					&next_hop);
			tbl = tbl_next;
		} while (status == 1);
		
		if (status < 0)
			next_hops[batch_index] = -1;
		else
			next_hops[batch_index] = next_hop;
	}
}

void rte_lpm6_lookup_goto(const struct rte_lpm6 *lpm,
                            uint8_t ips[][RTE_LPM6_IPV6_ADDR_SIZE],
                            int16_t *next_hops, unsigned n)
{
	const struct rte_lpm6_tbl_entry *tbl[BATCH_SIZE];
	const struct rte_lpm6_tbl_entry *tbl_next[BATCH_SIZE];
	uint32_t tbl24_index[BATCH_SIZE];
	uint8_t next_hop[BATCH_SIZE];
	uint8_t first_byte[BATCH_SIZE];
	int status[BATCH_SIZE];

	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&fpp_start;
	}

fpp_start:

        first_byte[I] = LOOKUP_FIRST_BYTE;
        tbl24_index[I] = (ips[I][0] << BYTES2_SIZE) |
        (ips[I][1] << BYTE_SIZE) | ips[I][2];
        FPP_PSS(&lpm->tbl24[tbl24_index[I]], fpp_label_1, n);
fpp_label_1:

        /* Calculate pointer to the first entry to be inspected */
        tbl[I] = &lpm->tbl24[tbl24_index[I]];
        
        do {
            /* Continue inspecting following levels until success or failure */
            status[I] = lookup_step(lpm, tbl[I], &tbl_next[I], ips[I], first_byte[I]++,
                                 &next_hop[I]);
            tbl[I] = tbl_next[I];
        } while (status[I] == 1);
        
        if (status[I] < 0)
            next_hops[I] = -1;
        else
            next_hops[I] = next_hop[I];
    
fpp_end:
    batch_rips[I] = &&fpp_end;
    iMask = FPP_SET(iMask, I); 
    if(iMask == (1 << BATCH_SIZE) - 1) {
        return;
    }
    I = (I + 1) & BATCH_SIZE_;
    goto *batch_rips[I];

}

/*
 * Finds a rule in rule table.
 * NOTE: Valid range for depth parameter is 1 .. 128 inclusive.
 */
static inline int32_t
rule_find(struct rte_lpm6 *lpm, uint8_t *ip, uint8_t depth)
{
	uint32_t rule_index;
	
	/* Scan used rules at given depth to find rule. */
	for (rule_index = 0; rule_index < lpm->used_rules; rule_index++) {
		/* If rule is found return the rule index. */
		if ((memcmp (lpm->rules_tbl[rule_index].ip, ip,
				RTE_LPM6_IPV6_ADDR_SIZE) == 0) &&
				lpm->rules_tbl[rule_index].depth == depth) {
		
			return rule_index;
		}
	}

	/* If rule is not found return -ENOENT. */
	return -ENOENT;
}

/*
 * Delete a rule from the rule table.
 * NOTE: Valid range for depth parameter is 1 .. 128 inclusive.
 */
static inline void
rule_delete(struct rte_lpm6 *lpm, int32_t rule_index)
{
	/*
	 * Overwrite redundant rule with last rule in group and decrement rule
	 * counter.
	 */
	lpm->rules_tbl[rule_index] = lpm->rules_tbl[lpm->used_rules-1];
	lpm->used_rules--;
}

/*
 * Deletes a rule
 */
int
rte_lpm6_delete(struct rte_lpm6 *lpm, uint8_t *ip, uint8_t depth)
{
	int32_t rule_to_delete_index;
	uint8_t ip_masked[RTE_LPM6_IPV6_ADDR_SIZE];
	unsigned i;
	
	/*
	 * Check input arguments.
	 */
	if ((lpm == NULL) || (depth < 1) || (depth > RTE_LPM6_MAX_DEPTH)) {
		return -EINVAL;
	}

	/* Copy the IP and mask it to avoid modifying user's input data. */
	memcpy(ip_masked, ip, RTE_LPM6_IPV6_ADDR_SIZE);
	mask_ip(ip_masked, depth);

	/*
	 * Find the index of the input rule, that needs to be deleted, in the
	 * rule table.
	 */
	rule_to_delete_index = rule_find(lpm, ip_masked, depth);

	/*
	 * Check if rule_to_delete_index was found. If no rule was found the
	 * function rule_find returns -ENOENT.
	 */
	if (rule_to_delete_index < 0)
		return rule_to_delete_index;

	/* Delete the rule from the rule table. */
	rule_delete(lpm, rule_to_delete_index);
		
	/* 
	 * Set all the table entries to 0 (ie delete every rule
	 * from the data structure.
	 */
	lpm->next_tbl8 = 0;
	memset(lpm->tbl24, 0, sizeof(lpm->tbl24));
	memset(lpm->tbl8, 0, sizeof(lpm->tbl8[0])
			* RTE_LPM6_TBL8_GROUP_NUM_ENTRIES * lpm->number_tbl8s);
	
	/* 
	 * Add every rule again (except for the one that was removed from
	 * the rules table).
	 */
	for (i = 0; i < lpm->used_rules; i++) {
		rte_lpm6_add(lpm, lpm->rules_tbl[i].ip, lpm->rules_tbl[i].depth,
				lpm->rules_tbl[i].next_hop);
	}
	
	return 0;
}

/*
 * Deletes a group of rules
 */
int
rte_lpm6_delete_bulk_func(struct rte_lpm6 *lpm,
		uint8_t ips[][RTE_LPM6_IPV6_ADDR_SIZE], uint8_t *depths, unsigned n)
{
	int32_t rule_to_delete_index;
	uint8_t ip_masked[RTE_LPM6_IPV6_ADDR_SIZE];
	unsigned i;
	
	/*
	 * Check input arguments.
	 */
	if ((lpm == NULL) || (ips == NULL) || (depths == NULL)) {
		return -EINVAL;
	}
	
	for (i = 0; i < n; i++) {
		/* Copy the IP and mask it to avoid modifying user's input data. */
		memcpy(ip_masked, ips[i], RTE_LPM6_IPV6_ADDR_SIZE);
		mask_ip(ip_masked, depths[i]);

		/*
		 * Find the index of the input rule, that needs to be deleted, in the
		 * rule table.
		 */
		rule_to_delete_index = rule_find(lpm, ip_masked, depths[i]);

		/*
		 * Check if rule_to_delete_index was found. If no rule was found the
		 * function rule_find returns -ENOENT.
		 */
		if (rule_to_delete_index < 0)
			continue;

		/* Delete the rule from the rule table. */
		rule_delete(lpm, rule_to_delete_index);
	}
		
	/* 
	 * Set all the table entries to 0 (ie delete every rule
	 * from the data structure.
	 */
	lpm->next_tbl8 = 0;
	memset(lpm->tbl24, 0, sizeof(lpm->tbl24));
	memset(lpm->tbl8, 0, sizeof(lpm->tbl8[0])
			* RTE_LPM6_TBL8_GROUP_NUM_ENTRIES * lpm->number_tbl8s);
	
	/* 
	 * Add every rule again (except for the ones that were removed from
	 * the rules table).
	 */
	for (i = 0; i < lpm->used_rules; i++) {
		rte_lpm6_add(lpm, lpm->rules_tbl[i].ip, lpm->rules_tbl[i].depth,
				lpm->rules_tbl[i].next_hop);
	}
	
	return 0;
}

/*
 * Delete all rules from the LPM table.
 */
void
rte_lpm6_delete_all(struct rte_lpm6 *lpm)
{
	/* Zero used rules counter. */
	lpm->used_rules = 0;
		
	/* Zero next tbl8 index. */
	lpm->next_tbl8 = 0;

	/* Zero tbl24. */
	memset(lpm->tbl24, 0, sizeof(lpm->tbl24));

	/* Zero tbl8. */
	memset(lpm->tbl8, 0, sizeof(lpm->tbl8[0]) *
			RTE_LPM6_TBL8_GROUP_NUM_ENTRIES * lpm->number_tbl8s);

	/* Delete all rules form the rules table. */
	memset(lpm->rules_tbl, 0, sizeof(struct rte_lpm6_rule) * lpm->max_rules);
}
