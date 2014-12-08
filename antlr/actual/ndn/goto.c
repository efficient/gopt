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
	char *name[BATCH_SIZE];
	int i[BATCH_SIZE];
	int c_i[BATCH_SIZE];
	int bkt_2[BATCH_SIZE];
	int bkt_1[BATCH_SIZE];
	int bkt_num[BATCH_SIZE];
	int terminate[BATCH_SIZE];
	int prefix_match_found[BATCH_SIZE];
	uint64_t prefix_hash[BATCH_SIZE];
	uint16_t tag[BATCH_SIZE];
	struct ndn_slot *slots[BATCH_SIZE];
	int8_t _dst_port[BATCH_SIZE];
	uint64_t _hash[BATCH_SIZE];

	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&fpp_start;
	}

fpp_start:

        name[I] = name_lo[I].name;
        FPP_PSS(name[I], fpp_label_1);
fpp_label_1:

         /**< URL char iterator and slot iterator */
        
        terminate[I] = 0;          /**< Stop processing this URL? */
        prefix_match_found[I] = 0; /**< Stop this hash-table lookup ? */
        
        /**< For names that we cannot find, dst_port is -1 */
        dst_ports[I] = -1;
        
        for(c_i[I] = 0; name[I][c_i[I]] != 0; c_i[I] ++) {
            if(name[I][c_i[I]] == '/') {
                break;
            }
        }
        
        c_i[I] ++;
        for(; name[I][c_i[I]] != 0; c_i[I] ++) {
            if(name[I][c_i[I]] != '/') {
                continue;
            }
            
            prefix_hash[I] = CityHash64WithSeed(name[I], c_i[I] + 1, NDN_SEED);
            tag[I] = prefix_hash[I] >> 48;
            
            /**< name[0] -> name[c_i] is a prefix of length c_i + 1 */
            for(bkt_num[I] = 1; bkt_num[I] <= 2; bkt_num[I] ++) {
                if(bkt_num[I] == 1) {
                    bkt_1[I] = prefix_hash[I] & NDN_NUM_BKT_;
                    FPP_PSS(&ht[bkt_1[I]], fpp_label_2);
fpp_label_2:

                    slots[I] = ht[bkt_1[I]].slots;
                } else {
                    bkt_2[I] = (bkt_1[I] ^ CityHash64((char *) &tag[I], 2)) & NDN_NUM_BKT_;
                    FPP_PSS(&ht[bkt_2[I]], fpp_label_3);
fpp_label_3:

                    slots[I] = ht[bkt_2[I]].slots;
                }
                
                /**< Now, "slots" points to an ndn_bucket. Find a valid slot
                 *  with a matching tag. */
                for(i[I] = 0; i[I] < NDN_NUM_SLOTS; i[I] ++) {
                    _dst_port[I] = slots[I][i[I]].dst_port;
                    _hash[I] = slots[I][i[I]].cityhash;
                    
                    if(_dst_port[I] >= 0 && _hash[I] == prefix_hash[I]) {
                        
                        /**< Record the dst port: this may get overwritten by
                         *  longer prefix matches later */
                        dst_ports[I] = slots[I][i[I]].dst_port;
                        
                        if(slots[I][i[I]].is_terminal == 1) {
                            /**< A terminal FIB entry: we're done! */
                            terminate[I] = 1;
                        }
                        
                        prefix_match_found[I] = 1;
                        break;
                    }
                }
                
                /**< Stop the hash-table lookup for name[0 ... c_i] */
                if(prefix_match_found[I] == 1) {
                    break;
                }
            }
            
            /**< Stop processing the name if we found a terminal FIB entry */
            if(terminate[I] == 1) {
                break;
            }
        }   /**< Loop over URL characters ends here */
        
       /**< Loop over batch ends here */

fpp_end:
    batch_rips[I] = &&fpp_end;
    iMask = FPP_SET(iMask, I); 
    if(iMask == (1 << BATCH_SIZE) - 1) {
        return;
    }
    I = (I + 1) & BATCH_SIZE_;
    goto *batch_rips[I];

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
