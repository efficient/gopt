import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class ProcRib {
	public static void main(String args[]) throws FileNotFoundException {
		int N = 19938;
		
		for(int reqLen = 0; reqLen <= 128; reqLen ++) {
			Scanner c = new Scanner(new File("uniq_ipv6_rib_201409"));
			for(int i = 0; i < N; i ++) {
				String line = c.nextLine();
				String prefix[] = line.split("/");
				
				// Compute the length of the prefix
				prefix[1] = prefix[1].substring(0, prefix[1].length());
				int prefixLen = Integer.parseInt(prefix[1]);
				if(prefixLen < 0 || prefixLen > 128) {
					System.out.println("Error");
					System.exit(-1);
				}
				
				if(prefixLen != reqLen) {
					continue;
				}
				
				String chunks[] = prefix[0].split(":");
				int numChunks = chunks.length;
				
				String res[] = new String[8];
				
				// Set all res's chunks to 0: make zero-compression easy 
				for(int j = 0; j < 8; j ++) {
					res[j] = "0";
				}
				
				boolean doReverse = false;
				for(int j = 0; j < numChunks; j ++) {
					if(chunks[j].length() == 0) {
						doReverse = true;
						break;
					}
					res[j] = chunks[j];
				}
				
				if(doReverse) {
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
					System.out.print(res[j] + " ");
					int resInt = Integer.parseInt(res[j], 16);
					if(resInt > 0xffff || resInt < 0) {
						System.out.println("Error");
						System.exit(-1);
					}
				}
				
				System.out.println(prefixLen);
			}
			c.close();
		}
	}
}

