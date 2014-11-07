#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<papi.h>
#include<time.h>

#include "city.h"
#include "fpp.h"
#include "ndn.h"

#define URL_FILE "/home/akalia/fastpp/data_dump/ndn_distributed_sample"
//#define URL_FILE "data/ndn_distributed_sample_small"
//#define URL_FILE "data/test"

int batch_index = 0;
int nb_succ = 0;

void process_batch(struct ndn_linear_url *url_lo, struct ndn_ht *ht) 
{
	foreach(batch_index, BATCH_SIZE) {

		char *url = url_lo[batch_index].url;

		int c_i, i;	/**< URL char iterator and slot iterator */
		int bkt_num, bkt_1, bkt_2;
		uint16_t tag = (uint16_t) url[0] + (((uint16_t) url[1]) << 8);

		struct ndn_bucket *ht_index = ht->ht_index;
		ULL *slot;
		uint8_t *ht_log = ht->ht_log;

		int match_found = 0;

		int url_len = strlen(url);
		url[url_len] = '/';		/**< This character was 0 */

		for(c_i = url_len; c_i >= 0; c_i --) {
			if(url[c_i] != '/') {
				continue;
			}

			/**< url[0] -> url[i] is a prefix of length i + 1 */
			for(bkt_num = 1; bkt_num <= 2; bkt_num ++) {
				if(bkt_num == 1) {
					bkt_1 = CityHash64(url, c_i + 1) & NDN_NUM_BKT_;
					FPP_EXPENSIVE(&ht_index[bkt_1]);
					slot = ht_index[bkt_1].slot;
				} else {
					bkt_2 = (bkt_1 ^ CityHash64((char *) &tag, 2)) & NDN_NUM_BKT_;
					FPP_EXPENSIVE(&ht_index[bkt_1]);
					slot = ht_index[bkt_2].slot;
				}

				/**< Now, "slot" points to an ndn_bucket. Find a valid slot 
				  *  with a matching tag. */
				for(i = 0; i < 8; i ++) {
					int slot_offset = NDN_SLOT_TO_OFFSET(slot[i]);
					uint16_t slot_tag = NDN_SLOT_TO_TAG(slot[i]);
					uint8_t *log_ptr = &ht_log[slot_offset];
	
					if(slot_offset != 0 && slot_tag == tag) {
						FPP_EXPENSIVE(log_ptr);
						uint8_t prefix_len = log_ptr[0];
						/**< Length of the current prefix is (i + 1) */
						if(prefix_len == (uint8_t) (c_i + 1) &&
							memcmp(url, &log_ptr[3], c_i + 1) == 0) {
							match_found = 1;
							nb_succ ++;
							break;
						}
					}
				}
	
				/**< Stop processing smaller prefixes if match found */
				if(match_found == 1) {
					break;
				}
			}

			/**< Stop processing buckets if match found */
			if(match_found == 1) {
				break;
			}
		}	/**< Loop over URL characters ends here */
	}	/**< Loop over batch ends here */
}

int main(int argc, char **argv)
{
	struct ndn_ht ht;
	int i;

	/** < Variables for PAPI */
	float real_time, proc_time, ipc;
	long long ins;
	int retval;

	red_printf("main: Initializing NDN hash table\n");
	ndn_init(URL_FILE, 0xf, &ht);
	red_printf("\tmain: Setting up NDN index done!\n");

	/*red_printf("main: Checking if all URLs were inserted\n");
	ndn_check(URL_FILE, &ht);
	red_printf("\tmain: Check succeeded\n");*/

	red_printf("main: Getting URL array\n");
	int nb_urls = ndn_get_num_urls(URL_FILE);
	nb_urls = nb_urls - (nb_urls % BATCH_SIZE);	/**< Align input to batch */

	struct ndn_linear_url *url_arr = ndn_get_url_array(URL_FILE);
	red_printf("\tmain: Constructed URL array!\n");

	red_printf("main: Starting NDN lookups\n");
	/** < Init PAPI_TOT_INS and PAPI_TOT_CYC counters */
	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	for(i = 0; i < nb_urls; i += BATCH_SIZE) {
		process_batch(&url_arr[i], &ht);
	}
	
	/**< All URLs should get a match */
	assert(nb_urls == nb_succ);

	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {    
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	red_printf("Time = %.4f s, rate = %.2f\n"
		"Instructions = %lld, IPC = %f\n", 
		real_time, nb_urls / real_time,
		ins, ipc);

	return 0;
}
