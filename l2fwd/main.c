#include "main.h"
int is_client = -1, client_id;

static struct ether_addr l2fwd_ports_eth_addr[RTE_MAX_ETHPORTS]; /**< MACs */
struct rte_mempool *l2fwd_pktmbuf_pool[RTE_MAX_LCORE];	/**< Per lcore mempools */

/**< Disable all offload features */
static const struct rte_eth_conf port_conf = {
	.rxmode = {
		.split_hdr_size = 0,
		.header_split   = 0,
		.hw_ip_checksum = 0,
		.hw_vlan_filter = 0,
		.jumbo_frame    = 0,
		.hw_strip_crc   = 0,
		.mq_mode = ETH_MQ_RX_RSS,
	},
    .rx_adv_conf = {
        .rss_conf = {
            .rss_key = NULL,
			.rss_hf = ETH_RSS_IPV4,
		},
	},
};

static const struct rte_eth_rxconf rx_conf = {
	.rx_thresh = {
		.pthresh = RX_PTHRESH,
		.hthresh = RX_HTHRESH,
		.wthresh = RX_WTHRESH,
	},
	.rx_free_thresh = DEFAULT_NIC_RX_FREE_THRESH,
	.rx_drop_en = 0		/**< No idea what this is */
};

static const struct rte_eth_txconf tx_conf = {
	.tx_thresh = {
		.pthresh = TX_PTHRESH,
		.hthresh = TX_HTHRESH,
		.wthresh = TX_WTHRESH,
	},
	.tx_free_thresh = 0, /* Use PMD default values */
	.tx_rs_thresh = 0, /* Use PMD default values */
};

static int
l2fwd_launch_one_lcore(__attribute__((unused)) void *dummy)
{
	if(is_client) {
		run_client(client_id, l2fwd_pktmbuf_pool);
	} else {
		run_server();
	}
	return 1;
}

int
main(int argc, char **argv)
{
	int ret;
	uint8_t nb_ports;
	uint8_t port_id;
	unsigned lcore_id;

	/**< Do args parsing before EAL's args parsing.
	  *  Do all data-structure hugepage allocations before EAL's init(). */
	if(argc > 5) {
		is_client = 1;
		client_id = atoi(argv[6]);
	} else {
		is_client = 0;
	}

	ret = rte_eal_init(argc, argv);
	CPE(ret < 0, "Invalid EAL arguments\n");

	CPE(rte_pmd_init_all() < 0, "Cannot init pmd\n");
	CPE(rte_eal_pci_probe() < 0, "Cannot probe PCI\n");

	nb_ports = rte_eth_dev_count();
	nb_ports = nb_ports > RTE_MAX_ETHPORTS ? RTE_MAX_ETHPORTS : nb_ports;
	CPE(nb_ports == 0, "No Ethernet ports - bye\n");

	printf("\n\n");

	/**< Create a mempool for each enabled lcore */
	for(lcore_id = 0; lcore_id < RTE_MAX_LCORE; lcore_id ++) {
		if(rte_lcore_is_enabled(lcore_id)) {
			char pool_name[20];
			sprintf(pool_name, "pool_%d", lcore_id);

			red_printf("Lcore %d is enabled. Creating mempool on socket %d\n",
				lcore_id, LCORE_TO_SOCKET(lcore_id));
			l2fwd_pktmbuf_pool[lcore_id] = mempool_init(pool_name,
				LCORE_TO_SOCKET(lcore_id));
			CPE(l2fwd_pktmbuf_pool[lcore_id] == NULL, "Cannot init mempool\n");
		}
	}

	/* Initialise each port */
	int portmask = is_client == 1 ? XIA_R0_PORT_MASK : XIA_R2_PORT_MASK;
	red_printf("\nInitializing ports\n");

	for (port_id = 0; port_id < nb_ports; port_id ++) {
		if (!ISSET(portmask, port_id)) {
			continue;
		}

		/**< xia-router0/1 use an IO-Hub for PCIe devices, so NICs don't have
		  *  a NUMA-socket. */
		int my_socket_id = is_client == 1 ?
			get_socket_id_from_macaddr(port_id) :
			rte_eth_dev_socket_id(port_id);

		/**< XXX: Need to implement logic so that server lcores only access
		  *  the ports on their socket. Until then, restrict to one socket */
		if(!is_client) {
			assert(my_socket_id == 0);
		}

		int num_queues = is_client == 1 ? 3 :
			count_active_lcores_on_socket(my_socket_id);

		printf("Initializing port %u on socket %d with %d queues \n", 
			(unsigned) port_id, my_socket_id, num_queues);

		ret = rte_eth_dev_configure(port_id,
			num_queues, num_queues, &port_conf);
		CPE2(ret < 0, "Cannot configure device: %d, %u\n",
			ret, (unsigned) port_id);

		int queue_id = 0;
		for(queue_id = 0; queue_id < num_queues; queue_id ++) {
			int my_lcore_id;
			if(is_client) {
				my_lcore_id = client_port_queue_to_lcore(port_id, queue_id);
			} else {
				my_lcore_id = get_lcore_ranked_n(queue_id, my_socket_id);
			}
	
			if(rte_lcore_is_enabled(my_lcore_id) == 0) {
				red_printf("\tQueue %d on port %d wants disabled lcore %d!\n",
					queue_id, port_id, my_lcore_id);
				exit(-1);
			}

			struct rte_mempool *mp = l2fwd_pktmbuf_pool[my_lcore_id];
			printf("\tSetting up queue %d using lcore %d's mempool\n",
				queue_id, my_lcore_id);

			ret = rte_eth_rx_queue_setup(port_id,
				queue_id, NUM_RX_DESC, my_socket_id, &rx_conf, mp);
			CPE2(ret < 0, "rte_eth_rx_queue_setup: %d, %u\n",
				ret, (unsigned) port_id);
	
			ret = rte_eth_tx_queue_setup(port_id,
				queue_id, NUM_TX_DESC, my_socket_id, &tx_conf);
			CPE2(ret < 0, "rte_eth_tx_queue_setup: %d, %u\n",
				ret, (unsigned) port_id);
		}

		/**< Print this port's MAC address and start it. */
		rte_eth_macaddr_get(port_id, &l2fwd_ports_eth_addr[port_id]);
		printf("Port %d, MAC: ", port_id);
		print_mac_arr(l2fwd_ports_eth_addr[port_id].addr_bytes);
		printf("\n");

		ret = rte_eth_dev_start(port_id);
		CPE2(ret < 0, "rte_eth_dev_start: %d, %u\n", ret, (unsigned) port_id);

	}

	check_all_ports_link_status(nb_ports, portmask);

	/**< Launch per-lcore init on every lcore */
	rte_eal_mp_remote_launch(l2fwd_launch_one_lcore, NULL, CALL_MASTER);
	RTE_LCORE_FOREACH_SLAVE(lcore_id) {
		if (rte_eal_wait_lcore(lcore_id) < 0)
			return -1;
	}
	return 0;
}

