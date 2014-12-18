import java.io.*;
import java.util.*;


/*< Class to generate prefixes and probe IPv6 addresses for testing */
public class ipv6TestGen {
	static int num_prefixes = 32;
	static int num_ips = 32;
	static int max_prefix_len = 48;
	private static Random generator = new Random(2);

	public static int rand() {
		return generator.nextInt(Integer.MAX_VALUE);
	}

	public static String getBits(byte b[], int len) {
		String ret = "";
		for(int i = 0; i < 16; i ++) {
			String temp = Integer.toString((int) b[i], 2);
			while(temp.length() != 8) {
				temp = "0" + temp;
			}
			ret = ret + temp;
		}

		return ret.substring(0, len);
	} 

	/*< An address matches a prefix (length = len) if its bit representation
	 *  up to len bits is same as the prefix */
	public static boolean match(IPv6Prefix p, IPv6Address a) {

		int len = p.len;

		String pBits = getBits(p.bytes, len);
		String aBits = getBits(a.bytes, len);
		boolean ret = pBits.equals(aBits);

		return ret;
	}

	public static void main(String args[]) {

		IPv6Prefix prefixes[] = new IPv6Prefix[num_prefixes];
		IPv6Address addresses[] = new IPv6Address[num_ips];

		/*< Generate the prefixes */
		System.out.println("Generating prefixes");

		for(int i = 0; i < num_prefixes; i ++) {
			prefixes[i] = new IPv6Prefix();

			/*< Length of this prefix in bits */
			prefixes[i].len = (rand() % max_prefix_len) + 1;
			System.out.print(prefixes[i].len + "  ");

			for(int j = 0; j < 16; j ++) {
				prefixes[i].bytes[j] = (byte) (rand() % 4);
				System.out.print(prefixes[i].bytes[j] + " ");
			}

			prefixes[i].dst_port = rand() % 256;
			System.out.println(" " + prefixes[i].dst_port);
		}

		/*< Generarate the probe IPs */
		System.out.println("Generating IPs");
		int dst_ports[] = new int[num_ips];

		for(int i = 0; i < num_ips; i ++) {
			addresses[i] = new IPv6Address();

			for(int j = 0; j < 16; j ++) {
				addresses[i].bytes[j] = (byte) (rand() % 4);
			}

			int dst_port = -1, lpm_len = -1;

			/*< Find the longest match among all prefixes */
			for(int j = 0; j < num_prefixes; j ++) {

				/*< Check if the current address matches this prefix */
				if(match(prefixes[j], addresses[i])) {
					int match_len = prefixes[j].len;

					if(match_len > lpm_len) {
						lpm_len = match_len;
						dst_port = prefixes[j].dst_port;
					}
				}
			}

			/*< If > 1 prefixes are LPM matches for this address, try a
			  * different address */
			int lpm_prefixes = 0;
			for(int j = 0; j < num_prefixes; j ++) {
				if(match(prefixes[j], addresses[i])) {
					if(prefixes[j].len == lpm_len) {
						lpm_prefixes ++;
					}
				}
			}

			if(lpm_prefixes > 1) {
				i --;
				continue;
			}
			

			for(int j = 0; j < 16; j ++) {
				System.out.print(addresses[i].bytes[j] + " ");
			}
			System.out.println();

			/*< Record the destination port for this IP; print later */
			dst_ports[i] = dst_port;
		}

		System.out.println("Expected output");
		for(int i = 0; i < num_ips; i ++) {
			System.out.println(dst_ports[i]);
		}
		
	}
}

class IPv6Prefix {
	byte bytes[];
	int len;
	int dst_port;

	public IPv6Prefix() {
		this.len = -1;
		this.dst_port = -1;
		this.bytes = new byte[16];
	}
}

class IPv6Address {
	byte bytes[];

	public IPv6Address() {
		this.bytes = new byte[16];
	}
}
