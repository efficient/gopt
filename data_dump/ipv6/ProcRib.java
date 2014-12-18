import java.io.*;
import java.util.*;

/*< Input: An IPv6 prefix file with zero-compressed IPv6 prefixes in
  * standard format. Examples of valid IPv6 addresses:
  * 2001:1200::/32, 2400:f000:80:8000:1::81/128.
  *
  * Output: Removes zero-compression from the prefixes. Prints:
  * byte_1, ..., byte_16, len for each IPv6 prefix */

public class ProcRib {
	public static void main(String args[]) throws FileNotFoundException {
		int N = 19938;
		
		/*< Only process prefixes of length = reqLen in this pass */
		for(int reqLen = 0; reqLen <= 128; reqLen ++) {
			Scanner c = new Scanner(new File("uniq_ipv6_rib_201409"));

			for(int i = 0; i < N; i ++) {
				String line = c.nextLine();
				String prefix[] = line.split("/");
				
				/* Check if the length of this prefix is reqLen */
				prefix[1] = prefix[1].substring(0, prefix[1].length());
				int prefixLen = Integer.parseInt(prefix[1]);
				if(prefixLen < 0 || prefixLen > 128) {
					System.out.println("Error");
					System.exit(-1);
				}
				
				if(prefixLen != reqLen) {
					continue;
				}
				
				String res[] = new String[8];	/*< The expanded prefix */
				for(int j = 0; j < 8; j ++) {
					res[j] = "0";
				}

				String chunks[] = prefix[0].split(":");
				int numChunks = chunks.length;
				
				/*< Copy the hexs upto the :: to res */
				boolean doReverse = false;
				for(int j = 0; j < numChunks; j ++) {

					if(chunks[j].length() == 0) {
						/*< This only happens if there are hexs after the ::
						 *  "x:y::".split(":") is ["x", "y"], but
						 *  "x:y::z".split(":") is ["x", "y", "", "z"] */
						doReverse = true;
						break;
					}
					res[j] = chunks[j];
				}

				if(doReverse) {
					/*< This means that there is an empty string in chunks[].
					 *  Copy chunks[] to res[] backwards till the null str */
					int k = 7;
					for(int j = numChunks - 1; j >= 0; j --) {
						if(chunks[j].length() == 0) {
							break;
						}
						res[k] = chunks[j];
						k --;
					}
				}
	
				for(int j = 0; j < 8; j ++) {

					int resInt = Integer.parseInt(res[j], 16);
					int msB = (resInt >> 8);
					int lsB = (resInt & 0xff);

					/*< Check if this IPv6 address is valid */
					if(resInt > 0xffff || resInt < 0 || msB >= 256 || msB < 0
						|| lsB >= 256 || lsB < 0) {
						System.out.println("Error: resInt = " + resInt +
							" msB = " + msB + " lsB = " + lsB);
						System.exit(-1);
					}

					System.out.print(msB + " " + lsB + " ");

				}
				
				System.out.println(prefixLen);
			}
			c.close();
		}
	}
}

