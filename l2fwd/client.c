#include "main.h"
#define MAX_CLT_TX_BURST 16
#define MAX_CLT_RX_BURST 16

void run_client(int client_id,
	struct rte_mempool **l2fwd_pktmbuf_pool, struct ipv6_prefix *prefix_arr)
{
	/**< [xia-router0 - xge0,1,2,3], [xia-router1 - xge0,1,2,3] */
	LL src_mac_arr[2][4] = {{0x36d3bd211b00, 0x37d3bd211b00, 0xa8d6a3211b00, 0xa9d6a3211b00},
							{0x44d7a3211b00, 0x45d7a3211b00, 0x0ad7a3211b00, 0x0bd7a3211b00}};

	/**< [xia-router2 - xge0,1,4,5], [xia-router2 - xge2,3,6,7] */
	LL dst_mac_arr[2][4] = {{0x6c10bb211b00, 0x6d10bb211b00, 0xc8a610ca0568, 0xc9a610ca0568},
							{0x64d2bd211b00, 0x65d2bd211b00, 0xa2a610ca0568, 0xa3a610ca0568}};

	/**< Even cores take xge0,1. Odd cores take xge2, xge3 */
	int lcore_to_port[12] = {0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3};

	int i;

	struct rte_mbuf *rx_pkts_burst[MAX_CLT_RX_BURST];
	struct rte_mbuf *tx_pkts_burst[MAX_CLT_TX_BURST];

	int lcore_id = rte_lcore_id();

	int port_id = lcore_to_port[lcore_id];
	if(!ISSET(XIA_R0_PORT_MASK, port_id)) {
		red_printf("Lcore %d uses disabled port (port %d). Exiting.\n",
			lcore_id, port_id);
		exit(-1);
	}

	/**< This is a valid queue_id because all client ports have 3 queues */
	int queue_id = lcore_id % 3;
	srand((int) rte_rdtsc());
	int prefix_arr_i = rand() % IPV6_NUM_RAND_PREFIXES;

	red_printf("Client: lcore = %d, port = %d, queue = %d, prfx_arr_i = %d \n",
		lcore_id, port_id, queue_id, prefix_arr_i);

	LL prev_tsc = 0, cur_tsc = 0;
	prev_tsc = rte_rdtsc();

	LL nb_tx = 0, nb_rx = 0;
	struct ether_hdr *eth_hdr;
	struct ipv6_hdr *ip6_hdr;
	struct ipv4_hdr *ip4_hdr;
	uint8_t *src_mac_ptr, *dst_mac_ptr;

	LL rx_samples = 0, latency_tot = 0;
	uint64_t rss_seed = 0xdeadbeef;

	/**< sizeof(ether_hdr) + sizeof(ipv6_hdr) is 54 --> 56 for 4 byte align */
	int hdr_size = 56;

	float sleep_us = 2;
	
	while (1) {

		for(i = 0; i < MAX_CLT_TX_BURST; i ++) {
			tx_pkts_burst[i] = rte_pktmbuf_alloc(l2fwd_pktmbuf_pool[lcore_id]);
			CPE(tx_pkts_burst[i] == NULL, "tx_alloc failed\n");
			
			eth_hdr = rte_pktmbuf_mtod(tx_pkts_burst[i], struct ether_hdr *);
			ip6_hdr = (struct ipv6_hdr *) ((char *) eth_hdr + sizeof(struct ether_hdr));
			ip4_hdr = (struct ipv4_hdr *) ((char *) eth_hdr + sizeof(struct ether_hdr));
		
			/**< Initialize Ethernet header */
			src_mac_ptr = &eth_hdr->s_addr.addr_bytes[0];
			if((fastrand(&rss_seed) & 0xff) == 0) {
				/**< STAMP: This pkt is used for GPU latency measurement. */
				set_mac(src_mac_ptr, src_mac_arr[client_id][port_id]);
			} else {
				set_mac(src_mac_ptr, 0xdeadbeef);
			}

			dst_mac_ptr = &eth_hdr->d_addr.addr_bytes[0];
			set_mac(dst_mac_ptr, dst_mac_arr[client_id][port_id]);

			/**< XXX HACK: Couldn't get IPv6 RSS working. */
			eth_hdr->ether_type = htons(ETHER_TYPE_IPv4);

			/**< Initialize IPv6 header */
			ip6_hdr->vtc_flow = 0;
			/**< 2B of padding after the v6 header, then TSC and magic */
			ip6_hdr->payload_len = 2 + sizeof(int) + sizeof(LL);
			ip6_hdr->proto = IPPROTO_IPV6;
			ip6_hdr->hop_limits = 64;

			/**< XXX HACK: Couldn't get IPv6 RSS working. */
			ip4_hdr->version_ihl = 0x40 | 0x05;
			
			/**< Put in the IPv6 addresses */
			rte_memcpy(ip6_hdr->dst_addr,
				prefix_arr[prefix_arr_i].bytes, IPV6_ADDR_LEN);
			rte_memcpy(ip6_hdr->src_addr,
				prefix_arr[prefix_arr_i].bytes, IPV6_ADDR_LEN);
			prefix_arr_i ++;
			if(prefix_arr_i == IPV6_NUM_RAND_PREFIXES) {
				prefix_arr_i = 0;
			}
			
			tx_pkts_burst[i]->pkt.nb_segs = 1;
			tx_pkts_burst[i]->pkt.pkt_len = hdr_size + ip6_hdr->payload_len;
			tx_pkts_burst[i]->pkt.data_len = hdr_size + ip6_hdr->payload_len;

			/**< Add client tsc */
			LL *clt_tsc = (LL *) (rte_pktmbuf_mtod(tx_pkts_burst[i], char *) +
				hdr_size);
			clt_tsc[0] = rte_rdtsc();	/**< 56 -> 64 */

			/**< Add global core-identifier, and timestamp */
			int *magic = (int *) (rte_pktmbuf_mtod(tx_pkts_burst[i], char *) + 
				hdr_size + sizeof(LL));
			magic[0] = client_id * 1000 + lcore_id;		/**< 64 -> 68 */
			
		}

		int nb_tx_new = rte_eth_tx_burst(port_id, 
			queue_id, tx_pkts_burst, MAX_CLT_TX_BURST);
		nb_tx += nb_tx_new;
		for(i = nb_tx_new; i < MAX_CLT_TX_BURST; i++) {
			rte_pktmbuf_free(tx_pkts_burst[i]);
		}

		micro_sleep(sleep_us, C_FAC);

		/**< RX drain */
		while(1) {
			int nb_rx_new = rte_eth_rx_burst(port_id, 
				queue_id, rx_pkts_burst, MAX_CLT_RX_BURST);
			if(nb_rx_new == 0) {
				break;
			}

			nb_rx += nb_rx_new;
			for(i = 0; i < nb_rx_new; i ++) {
				/**< Retrive send-TSC and lcore from which this pkt was sent */
				LL *clt_tsc = (LL *) (rte_pktmbuf_mtod(rx_pkts_burst[i], char *) +
					hdr_size);

				/**< Verify the server's response */
				int *magic = (int *) (rte_pktmbuf_mtod(rx_pkts_burst[i], char *) + 
					hdr_size + sizeof(LL));
				int tx_magic = magic[0];

				if(client_id * 1000 + lcore_id == tx_magic) {
					rx_samples ++;
					LL cur_tsc = rte_rdtsc();
					latency_tot += (cur_tsc - clt_tsc[0]);
				}

				rte_pktmbuf_free(rx_pkts_burst[i]);
			}
		}

		/**< Print TX stats : because clients rarely process RX pkts */
		if (unlikely(nb_tx >= 2000000)) {
			cur_tsc = rte_rdtsc();
			double nanoseconds = C_FAC * (cur_tsc - prev_tsc);
			prev_tsc = cur_tsc;

			printf("Lcore %d: TX = %.2f, latency = %.2f us, sleep = %.2f\n"
				"\tnb_rx = %lld, magic passed = %lld\n",
				lcore_id, nb_tx / (nanoseconds / GHZ_CPS),
				(C_FAC * (latency_tot / (rx_samples + .01))) / 1000, sleep_us,
				nb_rx, rx_samples);
			
			nb_tx = 0;

			nb_rx = 0;
			rx_samples = 0;
			latency_tot = 0;

			/**< Update sleep_us by reading the "sleep_time" file */
			sleep_us = get_sleep_time();
		}
	}
}
