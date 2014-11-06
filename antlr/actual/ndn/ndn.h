#include "util.h"

/**< SHM keys for the hash-table index and log */
#define NDN_HT_INDEX_KEY 1
#define NDN_HT_LOG_KEY 2

/**< The longest URL in data_dump/ndn_distributed_sample is 147 bytes*/
#define NDN_MAX_URL_LENGTH 150
#define NDN_LOG_HEADROOM (NDN_MAX_URL_LENGTH * 3)

/**< The maximum number of components is 5 */
#define NDN_MAX_COMPONENTS 5

/**< Don't want to include rte headers for RTE_MAX_ETHPORTS */
#define NDN_MAX_ETHPORTS 16
#define NDN_ISSET(a, i) (a & (1 << i))

/*************************************************************
These parameters are tuned for the ndn_distributed_sample file
in fastpp/data_dump. This file has 11 million URLs and is around
270 MB in size
**************************************************************/

/**< A URL is inserted into the hash index and the log multiple times
  *  (as many times as the number of components). So, the number of slots
  *  and the log size are large enough for a 3X overhead. */
#define NDN_NUM_BKT (8 * 1024 * 1024)
#define NDN_NUM_BKT_ (NDN_NUM_BKT - 1)

/**< Log entry format for an inserted prefix:
  *  byte 0: length of the prefix
  *  byte 1: == 1 iff this prefix is a terminal prefix 
  *  byte 2: destination port for this prefix
  *  byte 3 onwards: the actual prefix */
#define NDN_LOG_CAP (300 * 1024 * 1024)

/**< Slot: bytes 0:1 = tag | bytes 2:7 = offset in log */
struct ndn_bucket
{
	ULL slot[8];
};

struct ndn_ht
{
	struct ndn_bucket *ht_index;
	uint8_t *ht_log;
	ULL log_head;	/**< log_head >= 1 means that this slot is valid */
};

/**< For storing URLs linearly */
struct ndn_linear_url
{
	char url[NDN_MAX_URL_LENGTH];
};

/**< These macros should be safe for use with the ANTLR code */
#define NDN_SLOT_TO_OFFSET(s) (s & ((1L << 48) - 1))	/**< Lower 48 bytes */
#define NDN_SLOT_TO_TAG(s) (s >> 48)	/**< Higher 16 bytes */

/**< NDN-specific function prototypes */
void ndn_init(const char *urls_file, int portmask, struct ndn_ht *ht);

/**< Insert a prefix (specified by "url" and "len") into the NDN hash table. 
  *  Returns 0 on success and -1 on failure. */
int ndn_ht_insert(const char *url, int len, 
	int is_terminal, int dst_port_id, struct ndn_ht *ht);

/**< Check if all the URLs in "urls_file" are inserted in the hash table */
void ndn_check(const char *urls_file, struct ndn_ht *ht);

/**< Return the number of URLs in a file */
int ndn_get_num_urls(const char *urls_file);

/**< Put all the URLs in a linear array with fixed sized slots */
struct ndn_linear_url *ndn_get_url_array(const char *urls_file);

/**< Print some useful stats for the URLs in this file */
void ndn_print_url_stats(const char *urls_file);
