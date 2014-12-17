import java.io.*;
import java.util.*;


/*< Class to generate prefixes and probe IPv6 addresses for testing */
public class ipv6TestGen {
	static int num_prefixes = 32;
	static int num_ips = 8;
	static int max_prefix_len = 48;

	public static int rand() {
		return (int) (Math.random() * 1000000000);
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

	public static boolean match(IPv6Prefix p, IPv6Address a) {
		String pBytes = Arrays.toString(p.bytes);
		String aBytes = Arrays.toString(a.bytes);


		int len = p.len;
		String pBits = getBits(p.bytes, len);
		String aBits = getBits(a.bytes, len);
		boolean ret = pBits.equals(aBits);

		System.out.printf("Matching prefix %s, %d with address %s --> %B\n",
			pBytes, p.len, aBytes, ret);
	
		return ret;
	}

	public static void main(String args[]) {

		IPv6Prefix prefixes[] = new IPv6Prefix[num_prefixes];
		IPv6Address addresses[] = new IPv6Address[num_ips];

		/*< Generate the prefixes */
		for(int i = 0; i < num_prefixes; i ++) {
			prefixes[i] = new IPv6Prefix();

			/*< Length of this prefix in bits */
			prefixes[i].len = (rand() % max_prefix_len) + 1;
			System.out.print(prefixes[i].len + " ");

			for(int j = 0; j < 16; j ++) {
				prefixes[i].bytes[j] = (byte) (rand() % 4);
				System.out.print(prefixes[i].bytes[j] + " ");
			}

			prefixes[i].dst_port = rand() % 8;
			System.out.println(prefixes[i].dst_port);
		}

		/*< Generarate the probe IPs */
		for(int i = 0; i < num_ips; i ++) {
			addresses[i] = new IPv6Address();

			for(int j = 0; j < 16; j ++) {
				addresses[i].bytes[j] = (byte) (rand() % 4);
				System.out.print(addresses[i].bytes[j] + " ");
			}

			int dst_port = -1, longest_match = -1;

			/*< Find the longest match among all prefixes */
			for(int j = 0; j < num_prefixes; j ++) {

				/*< Check if the current address matches this prefix */
				if(match(prefixes[j], addresses[i])) {
					int match_len = prefixes[j].len;

					if(match_len > longest_match) {
						longest_match = match_len;
						dst_port = prefixes[j].dst_port;
					}
				}
			}

			System.out.println(dst_port);
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
