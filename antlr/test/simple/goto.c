// Process BATCH_SIZE pkts starting from lo
#include "fpp.h"
int process_pkts_in_batch(int *pkt_lo)
{
	int mem_addr[BATCH_SIZE];

	int I = 0;			// batch index
	void *batch_rips[BATCH_SIZE];		// goto targets
	int iMask = 0;		// No packet is done yet

	int temp_index;
	for(temp_index = 0; temp_index < BATCH_SIZE; temp_index ++) {
		batch_rips[temp_index] = &&label_0;
	}

label_0:

    // Like a foreach loop
    
        mem_addr[I] = hash(pkt_lo[I]) & LOG_CAP_;
		FPP_PSS(&ht_log[mem_addr[I]], label_1);
label_1:

        sum += ht_log[mem_addr[I]];
    
end:
    batch_rips[I] = &&end;
    iMask = FPP_SET(iMask, I); 
    if(iMask == (1 << BATCH_SIZE) - 1) {
        return;
    }
    I = (I + 1) & BATCH_SIZE_;
    goto *batch_rips[I];

}

