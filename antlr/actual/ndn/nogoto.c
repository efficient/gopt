#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<papi.h>
#include<time.h>

#include "city.h"
#include "fpp.h"
#include "ndn.h"

int batch_index = 0;

void process_batch(struct ndn_name *name_lo, int *dst_ports,
	struct ndn_bucket *ht) 
{
	foreach(batch_index, BATCH_SIZE) {
		char *name = name_lo[batch_index].name;
		FPP_EXPENSIVE(name);

		int c_i, i;	/**< URL char iterator and slot iterator */
		int bkt_num, bkt_1, bkt_2;

		int terminate = 0;			/**< Stop processing this URL? */
		int prefix_match_found = 0;	/**< Stop this hash-table lookup ? */

		/**< For names that we cannot find, dst_port is -1 */
		dst_ports[batch_index] = -1;

		for(c_i = 0; name[c_i] != 0; c_i ++) {
			if(name[c_i] == '/') {
				break;
			}
		}

		c_i ++;
		for(; name[c_i] != 0; c_i ++) {
			if(name[c_i] != '/') {
				continue;
			}

			uint64_t prefix_hash = CityHash64WithSeed(name, c_i + 1, NDN_SEED);
			uint16_t tag = prefix_hash >> 48;

			struct ndn_slot *slots;

			/**< name[0] -> name[c_i] is a prefix of length c_i + 1 */
			for(bkt_num = 1; bkt_num <= 2; bkt_num ++) {
				if(bkt_num == 1) {
					bkt_1 = prefix_hash & NDN_NUM_BKT_;
					FPP_EXPENSIVE(&ht[bkt_1]);
					slots = ht[bkt_1].slots;
				} else {
					bkt_2 = (bkt_1 ^ CityHash64((char *) &tag, 2)) & NDN_NUM_BKT_;
					FPP_EXPENSIVE(&ht[bkt_2]);
					slots = ht[bkt_2].slots;
				}

				/**< Now, "slots" points to an ndn_bucket. Find a valid slot
				  *  with a matching tag. */
				for(i = 0; i < NDN_NUM_SLOTS; i ++) {
					int8_t _dst_port = slots[i].dst_port;
					uint64_t _hash = slots[i].cityhash;

					if(_dst_port >= 0 && _hash == prefix_hash) {

						/**< Record the dst port: this may get overwritten by
						  *  longer prefix matches later */
						dst_ports[batch_index] = slots[i].dst_port;

						if(slots[i].is_terminal == 1) {
							/**< A terminal FIB entry: we're done! */
							terminate = 1;
						}

						prefix_match_found = 1;
						break;
					}
				}

				/**< Stop the hash-table lookup for name[0 ... c_i] */
				if(prefix_match_found == 1) {
					break;
				}
			}

			/**< Stop processing the name if we found a terminal FIB entry */
			if(terminate == 1) {
				break;
			}
		}	/**< Loop over URL characters ends here */
	
	}	/**< Loop over batch ends here */
}

int main(int argc, char **argv)
{
	printf("%lu\n", sizeof(struct ndn_bucket));
	struct ndn_bucket *ht;
	int i, j;
	int dst_ports[BATCH_SIZE], nb_succ = 0, dst_port_sum = 0;

	/** < Variables for PAPI */
	float real_time, proc_time, ipc;
	long long ins;
	int retval;

	red_printf("main: Initializing NDN hash table\n");
	ndn_init(URL_FILE, 0xf, &ht);
	red_printf("\tmain: Setting up NDN index done!\n");

	red_printf("main: Getting name array for lookups\n");
	int nb_names = ndn_get_num_lines(NAME_FILE);
	nb_names = nb_names - (nb_names % BATCH_SIZE);	/**< Align input to batch */

	struct ndn_name *name_arr = ndn_get_name_array(NAME_FILE);
	red_printf("\tmain: Constructed name array!\n");

	red_printf("main: Starting NDN lookups\n");

	/** < Init PAPI_TOT_INS and PAPI_TOT_CYC counters */
	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	for(i = 0; i < nb_names; i += BATCH_SIZE) {
		memset(dst_ports, -1, BATCH_SIZE * sizeof(int));
		process_batch(&name_arr[i], dst_ports, ht);

		for(j = 0; j < BATCH_SIZE; j ++) {
			#if NDN_DEBUG == 1
			printf("Name %s -> port %d\n", name_arr[i + j].name, dst_ports[j]);
			#endif
			nb_succ += (dst_ports[j] == -1) ? 0 : 1;
			dst_port_sum += dst_ports[j];
		}
	}

	if((retval = PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK) {
		printf("PAPI error: retval: %d\n", retval);
		exit(1);
	}

	red_printf("Time = %.4f s, Lookup rate = %.2f M/s | nb_succ = %d, sum = %d\n"
		"Instructions = %lld, IPC = %f\n",
		real_time, nb_names / (real_time * 1000000), nb_succ, dst_port_sum,
		ins, ipc);

	return 0;
}
