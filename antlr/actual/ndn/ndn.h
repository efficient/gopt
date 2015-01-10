#include "util.h"

#define NDN_DEBUG 0

#if NDN_DEBUG == 1
#define URL_FILE "../../../data_dump/ndn/fib_test"
#define NAME_FILE "../../../data_dump/ndn/name_test"
#else
#define URL_FILE "../../../data_dump/ndn/fib_1010"
#define NAME_FILE "../../../data_dump/ndn/fib_1010"
#endif

#define NDN_SEED 3185

/**< SHM key for the hash-table index */
#define NDN_HT_INDEX_KEY 1

/**< SHM key for for the names array */
#define NDN_NAMES_KEY 2

/**< The longest URL in Tsinghua's 10M FIB is 97 bytes*/
#define NDN_MAX_URL_LENGTH 150
#define NDN_LOG_HEADROOM (NDN_MAX_URL_LENGTH * 3)

#define NDN_MAX_NAME_LENGTH 192		/**< For struct ndn_name */
#define NDN_MAX_LINE_LENGTH 10000	/**< For ndn_get_num_lines() */

/**< The maximum number of components is Tsinghua's 10M FIB is 13 */
#define NDN_MAX_COMPONENTS 15

/**< Don't want to include rte headers for RTE_MAX_ETHPORTS */
#define NDN_MAX_ETHPORTS 16
#define NDN_ISSET(a, i) (a & (1 << i))

/*************************************************************
These parameters are tuned for the ndn_distributed_sample file
in fastpp/data_dump. This file has 11 million URLs and is around
270 MB in size
**************************************************************/

/**< A URL is inserted into the hash index multiple times (as many times
  *  as the number of components). So, the number of slots is large enough
  *  for a 3X overhead. */
#define NDN_NUM_BKT (8 * 1024 * 1024)
#define NDN_NUM_BKT_ (NDN_NUM_BKT - 1)
#define NDN_NUM_SLOTS 8

struct __attribute__((__packed__)) ndn_slot
{
	int8_t dst_port;			/**< -1 for invalid slots */
	uint8_t is_terminal;
	uint64_t cityhash;
	uint8_t pad[6];				/**< Space for more signatures */
};

struct ndn_bucket
{
	struct ndn_slot slots[NDN_NUM_SLOTS];
};

/**< For storing URLs linearly */
struct ndn_name
{
	char name[NDN_MAX_NAME_LENGTH];
};

/**< These macros should be safe for use with the ANTLR code */
#define NDN_SLOT_TO_OFFSET(s) (s & ((1L << 48) - 1))	/**< Lower 48 bytes */
#define NDN_SLOT_TO_TAG(s) (s >> 48)	/**< Higher 16 bytes */

/**< Fast crc using SSE instructions */
uint32_t ndn_crc(const char *str, uint32_t len);

/**< NDN-specific function prototypes */
void ndn_init(const char *urls_file, int portmask, struct ndn_bucket **ht);

/**< Insert a prefix (specified by "url" and "len") into the NDN hash table. 
  *  Returns 0 on success and -1 on failure. */
int ndn_ht_insert(const char *url, int len, 
	int is_terminal, int dst_port_id, struct ndn_bucket *ht);

/**< Check if all the URLs in "urls_file" are inserted in the hash table */
void ndn_check(const char *urls_file, struct ndn_bucket *ht);

/**< Return the number of lines in a file */
int ndn_get_num_lines(const char *file_name);

/**< Put all the URLs in a linear array with fixed sized slots */
struct ndn_name *ndn_get_name_array(const char *names_file);

/**< Print some useful stats for the URLs in this file */
void ndn_print_url_stats(const char *urls_file);

/**< Count the number of components in this URL */
inline int ndn_num_components(const char *url);

/**< Create a mutable prefix from a URL */
char *ndn_get_prefix(const char *url, int len);

