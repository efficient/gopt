import java.io.*;
import java.util.*;

/*< Input: An IPv4 prefix file with IPv4 prefixes in standard format.
  * Example of valid IPv4 address:
  * 	PREFIX: 100.0.0.0/16
  *
  * Prints: depth  byte_1 ... byte_16  dst_port for each IPv4 prefix */

public class ProcRib {
	public static void main(String args[]) throws FileNotFoundException {
		
		Random randGen = new Random(2);

		int N = 527961;
		Scanner c = new Scanner(new File("uniq_ipv4_rib_201409"));
		int stats[] = new int[33];

		for(int i = 0; i < N; i ++) {
			String line = c.nextLine();
			String prefix = line.split(" ")[1];

			String bytes[] = prefix.split("/");

			String prefixBytes = bytes[0].replace(".", " ");
			int depth = Integer.parseInt(bytes[1]);
			stats[depth] ++;

			int dstPort = randGen.nextInt(256);

			System.out.println(depth + "  " + prefixBytes + "  " + dstPort);
		}

		for(int i = 0; i <= 32; i ++) {
			System.err.println("Prefixes with len = " + i + " = " + stats[i]);
			System.err.println("\tPossible IPv4 addresses = " + stats[i] * (1 << (32 - i)));
		}

		c.close();
	}
}

