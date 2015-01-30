// Defs for reference. XXX: This header file is unused.

/**
 * A packet message buffer.
 */
struct rte_pktmbuf {
	/* valid for any segment */
	struct rte_mbuf *next;  /**< Next segment of scattered packet. */
	void* data;             /**< Start address of data in segment buffer. */
	uint16_t data_len;      /**< Amount of data in segment buffer. */

	/* these fields are valid for first segment only */
	uint8_t nb_segs;        /**< Number of segments. */
	uint8_t in_port;        /**< Input port. */
	uint32_t pkt_len;       /**< Total pkt len: sum of all segment data_len. */

	/* offload features */
	union rte_vlan_macip vlan_macip;
	union {
		uint32_t rss;       /**< RSS hash result if RSS enabled */
		struct {
			uint16_t hash;
			uint16_t id;
		} fdir;             /**< Filter identifier if FDIR enabled */
		uint32_t sched;     /**< Hierarchical scheduler */
	} hash;                 /**< hash information */
};

/**
 * The generic rte_mbuf, containing a packet mbuf or a control mbuf.
 */
struct rte_mbuf {
	struct rte_mempool *pool; /**< Pool from which mbuf was allocated. */
	void *buf_addr;           /**< Virtual address of segment buffer. */
	phys_addr_t buf_physaddr; /**< Physical address of segment buffer. */
	uint16_t buf_len;         /**< Length of segment buffer. */
#ifdef RTE_MBUF_SCATTER_GATHER
	/**
	 * 16-bit Reference counter.
	 * It should only be accessed using the following functions:
	 * rte_mbuf_refcnt_update(), rte_mbuf_refcnt_read(), and
	 * rte_mbuf_refcnt_set(). The functionality of these functions (atomic,
	 * or non-atomic) is controlled by the CONFIG_RTE_MBUF_REFCNT_ATOMIC
	 * config option.
	 */
	union {
		rte_atomic16_t refcnt_atomic;   /**< Atomically accessed refcnt */
		uint16_t refcnt;                /**< Non-atomically accessed refcnt */
	};
#else
	uint16_t refcnt_reserved;     /**< Do not use this field */
#endif
	uint8_t type;                 /**< Type of mbuf. */
	uint8_t reserved;             /**< Unused field. Required for padding. */
	uint16_t ol_flags;            /**< Offload features. */

	union {
		struct rte_ctrlmbuf ctrl;
		struct rte_pktmbuf pkt;
	};
} __rte_cache_aligned;

struct ether_hdr {
	struct ether_addr d_addr; /**< Destination address. */
	struct ether_addr s_addr; /**< Source address. */
	uint16_t ether_type;      /**< Frame type. */
} __attribute__((__packed__));

/**
 * Ethernet address:
 * A universally administered address is uniquely assigned to a device by its
 * manufacturer. The first three octets (in transmission order) contain the
 * Organizationally Unique Identifier (OUI). The following three (MAC-48 and
 * EUI-48) octets are assigned by that organization with the only constraint
 * of uniqueness.
 * A locally administered address is assigned to a device by a network
 * administrator and does not contain OUIs.
 * See http://standards.ieee.org/regauth/groupmac/tutorial.html
 */
struct ether_addr {
	uint8_t addr_bytes[ETHER_ADDR_LEN]; /**< Address bytes in transmission order */
} __attribute__((__packed__));

/*
 * IPv4 Header: size = 20 bytes
 */
struct ipv4_hdr {
    uint8_t  version_ihl;       /**< version and header length */
    uint8_t  type_of_service;   /**< type of service */
    uint16_t total_length;      /**< length of packet */
    uint16_t packet_id;     /**< packet ID */
    uint16_t fragment_offset;   /**< fragmentation offset */
    uint8_t  time_to_live;      /**< time to live */
    uint8_t  next_proto_id;     /**< protocol ID */
    uint16_t hdr_checksum;      /**< header checksum */
    uint32_t src_addr;      /**< source address */
    uint32_t dst_addr;      /**< destination address */
} __attribute__((__packed__));

/**
 * IPv6 Header: size = 40 bits
 */
struct ipv6_hdr {
    uint32_t vtc_flow;     /**< IP version, traffic class & flow label. */
    uint16_t payload_len;  /**< IP packet length - includes sizeof(ip_header). */
    uint8_t  proto;        /**< Protocol, next header. */
    uint8_t  hop_limits;   /**< Hop limits. */
    uint8_t  src_addr[16]; /**< IP address of source host. */
    uint8_t  dst_addr[16]; /**< IP address of destination host(s). */
} __attribute__((__packed__));

