// Process BATCH_SIZE pkts starting from lo
int process_pkts_in_batch(int *pkt_lo)
{
	// Like a foreach loop
	foreach(batch_index, BATCH_SIZE) {

		int a_1 = hash(pkt_lo[batch_index]) & LOG_CAP_;
		int a_2 = hash(a_1) & LOG_CAP_;
		int a_3 = hash(a_2) & LOG_CAP_;
		int a_4 = hash(a_3) & LOG_CAP_;
		int a_5 = hash(a_4) & LOG_CAP_;
		int a_6 = hash(a_5) & LOG_CAP_;
		int a_7 = hash(a_6) & LOG_CAP_;
		int a_8 = hash(a_7) & LOG_CAP_;
		int a_9 = hash(a_8) & LOG_CAP_;
		int a_10 = hash(a_9) & LOG_CAP_;
		int a_11 = hash(a_10) & LOG_CAP_;
		int a_12 = hash(a_11) & LOG_CAP_;
		int a_13 = hash(a_12) & LOG_CAP_;
		int a_14 = hash(a_13) & LOG_CAP_;
		int a_15 = hash(a_14) & LOG_CAP_;
		int a_16 = hash(a_15) & LOG_CAP_;
		int a_17 = hash(a_16) & LOG_CAP_;
		int a_18 = hash(a_17) & LOG_CAP_;
		int a_19 = hash(a_18) & LOG_CAP_;
		int a_20 = hash(a_19) & LOG_CAP_;
		
		PREFETCH(&ht_log[a_20]);
		sum += ht_log[a_20];
	}
}

