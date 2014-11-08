#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<papi.h>
#include<time.h>

#include "city.h"
#include "fpp.h"
#include "ndn.h"

int batch_index = 0;
int nb_succ = 0;

void process_batch(struct ndn_linear_url *url_lo, struct ndn_ht *ht)
{
	char *url[BATCH_SIZE];
	int i[BATCH_SIZE];
	int c_i[BATCH_SIZE];
	int bkt_2[BATCH_SIZE];
	int bkt_1[BATCH_SIZE];
	int bkt_num[BATCH_SIZE];
	uint16_t tag[BATCH_SIZE];
	struct ndn_bucket *ht_index[BATCH_SIZE];
	ULL *slot[BATCH_SIZE];
	uint8_t *ht_log[BATCH_SIZE];
	int match_found[BATCH_SIZE];
	int url_len[BATCH_SIZE];
	int slot_offset[BATCH_SIZE];
	uint16_t slot_tag[BATCH_SIZE];
	uint8_t *log_ptr[BATCH_SIZE];
	uint8_t prefix_len[BATCH_SIZE];

	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = && fpp_start;
	}

fpp_start:

	url[I] = url_lo[I].url;
	url_len[I] = strlen(url[I]);

	/**< The last character (not '/') is a good tag */
	tag[I] = url[I][url_len[I] - 1];
	url[I][url_len[I]] = '/';		/**< This character was 0 */

	/**< URL char iterator and slot iterator */

	ht_index[I] = ht->ht_index;

	ht_log[I] = ht->ht_log;

	match_found[I] = 0;


	for(c_i[I] = url_len[I]; c_i[I] >= 0; c_i[I] --) {
		if(url[I][c_i[I]] != '/') {
			continue;
		}

		/**< url[0] -> url[i] is a prefix of length i + 1 */
		for(bkt_num[I] = 1; bkt_num[I] <= 2; bkt_num[I] ++) {
			if(bkt_num[I] == 1) {
				bkt_1[I] = CityHash64(url[I], c_i[I] + 1) & NDN_NUM_BKT_;
				FPP_PSS(&ht_index[I][bkt_1[I]], fpp_label_1);
fpp_label_1:

				slot[I] = ht_index[I][bkt_1[I]].slot;
			} else {
				bkt_2[I] = (bkt_1[I] ^ CityHash64((char *) &tag[I], 2)) & NDN_NUM_BKT_;
				FPP_PSS(&ht_index[I][bkt_1[I]], fpp_label_2);
fpp_label_2:

				slot[I] = ht_index[I][bkt_2[I]].slot;
			}

			/**< Now, "slot" points to an ndn_bucket. Find a valid slot
			  *  with a matching tag. */
			for(i[I] = 0; i[I] < 8; i[I] ++) {
				slot_offset[I] = NDN_SLOT_TO_OFFSET(slot[I][i[I]]);
				slot_tag[I] = NDN_SLOT_TO_TAG(slot[I][i[I]]);
				log_ptr[I] = &ht_log[I][slot_offset[I]];

				if(slot_offset[I] != 0 && slot_tag[I] == tag[I]) {
					FPP_PSS(log_ptr[I], fpp_label_3);
fpp_label_3:

					prefix_len[I] = log_ptr[I][0];
					/**< Length of the current prefix is (i + 1) */
					if(prefix_len[I] == (uint8_t) (c_i[I] + 1) &&
					        memcmp(url[I], &log_ptr[I][3], c_i[I] + 1) == 0) {
						match_found[I] = 1;
						nb_succ ++;
						break;
					}
				}
			}

			/**< Stop processing smaller prefixes if match found */
			if(match_found[I] == 1) {
				break;
			}
		}

		/**< Stop processing buckets if match found */
		if(match_found[I] == 1) {
			break;
		}
	}	/**< Loop over URL characters ends here */
	/**< Loop over batch ends here */

fpp_end:
	batch_rips[I] = && fpp_end;
	iMask = FPP_SET(iMask, I);
	if(iMask == (1 << BATCH_SIZE) - 1) {
		return;
	}
	I = (I + 1) & BATCH_SIZE_;
	goto *batch_rips[I];

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
