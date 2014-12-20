import java.io.*;
import java.util.*;

/*< Class to generate prefixes and probe IPv4 addresses for testing */
public class ipv4TestGen {
	static int num_prefixes = 1024;
	static int num_ips = 1024;

	static int BYTE_MOD = 4;

	private static Random generator = new Random(2);

	public static int rand() {
		return generator.nextInt(Integer.MAX_VALUE);
	}

	public static String getBits(byte b[], int len) {
		String ret = "";
		for(int i = 0; i < 4; i ++) {
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
	public static boolean match(IPv4Prefix p, IPv4Address a) {

		int len = p.len;

		String pBits = getBits(p.bytes, len);
		String aBits = getBits(a.bytes, len);
		boolean ret = pBits.equals(aBits);

		return ret;
	}

	public static void main(String args[]) {

		IPv4Prefix prefixes[] = new IPv4Prefix[num_prefixes];
		IPv4Address addresses[] = new IPv4Address[num_ips];

		/*< Generate the prefixes */
		System.out.println(num_prefixes);

		for(int i = 0; i < num_prefixes; i ++) {
			prefixes[i] = new IPv4Prefix();

			/*< Length of this prefix in bits */
			prefixes[i].len = 24;
			if(rand() % 10 == 0) {
				prefixes[i].len --;
			} else if(rand() % 10 <= 1) {
				prefixes[i].len ++;
			}

			System.out.print(prefixes[i].len + "  ");

			for(int j = 0; j < 4; j ++) {
				prefixes[i].bytes[j] = (byte) (rand() % BYTE_MOD);
				System.out.print(prefixes[i].bytes[j] + " ");
			}

			prefixes[i].dst_port = rand() % 256;
			System.out.println(" " + prefixes[i].dst_port);
		}

		/*< Generarate the probe IPs */
		System.err.println(num_ips);
		int dst_ports[] = new int[num_ips];

		for(int i = 0; i < num_ips; i ++) {
			addresses[i] = new IPv4Address();

			for(int j = 0; j < 4; j ++) {
				addresses[i].bytes[j] = (byte) (rand() % BYTE_MOD);
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
			

			for(int j = 0; j < 4; j ++) {
				System.err.print(addresses[i].bytes[j] + " ");
			}
			System.err.println();

			/*< Record the destination port for this IP; print later */
			dst_ports[i] = dst_port;
		}

		for(int i = 0; i < num_ips; i ++) {
			System.err.println(dst_ports[i]);
		}
		
	}
}

class IPv4Prefix {
	byte bytes[];
	int len;
	int dst_port;

	public IPv4Prefix() {
		this.len = -1;
		this.dst_port = -1;
		this.bytes = new byte[4];
	}
}

class IPv4Address {
	byte bytes[];

	public IPv4Address() {
		this.bytes = new byte[4];
	}
}
