static void cuckoo_forward_burst_2(struct lcore_params *lp,
									struct rte_mbuf *m1,
									struct rte_mbuf *m2,
									uint8_t port,
									uint32_t socket)
{
	struct ether_hdr *eth;
	uint32_t i;
	uint64_t m1_ether_addr = 0, m2_ether_addr = 0, m3_ether_addr = 0, m4_ether_addr = 0;
	uint16_t m1_outport, m2_outport, m3_outport, m4_outport;

	for (i = 0; i < 6; i++)
		m1_ether_addr = (m1_ether_addr << 8) | (eth->d_addr.addr_bytes[i]);
	for (i = 0; i < 6; i++)
		m2_ether_addr = (m2_ether_addr << 8) | (eth->d_addr.addr_bytes[i]);

	cuckoo_status st = cuckoo_find_burst_2(h,
						m1_ether_addr, &m1_outport,
						m2_ether_addr, &m2_outport);
	assert(st == ok);

	m1_outport = (m1_outport & 3) + socket * 4;
	m2_outport = (m2_outport & 3) + socket * 4;


	send_packet(lp, m1, m1_outport);
	send_packet(lp, m2, m2_outport);
}
