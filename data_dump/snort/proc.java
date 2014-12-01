import java.util.*;
import java.io.*;
import java.lang.*;

public class proc {
	public static void main(String a[]) throws Exception {
		BufferedReader c1 = new BufferedReader(new FileReader("snort_dfa_patterns"));
		BufferedReader c2 = new BufferedReader(new FileReader("snort_packets"));
		PrintWriter out = new PrintWriter(System.out);

		Map<String, Integer> dfaMap = new HashMap<String, Integer>();
		int dfa_id = 0;
		while(true) {
			String insertedPattern = c1.readLine();
			if(insertedPattern == null) {
				break;
			}
			String parts[] = insertedPattern.split(" ");
			if(!dfaMap.containsKey(parts[0])) {
				dfaMap.put(parts[0], dfa_id);
				System.err.println(parts[0] + " assigned ID " + dfa_id);
				dfa_id ++;
			}

			parts[0] = "" + dfaMap.get(parts[0]);
			for(int i = 0; i < parts.length; i ++) {
				out.print(parts[i] + " ");
			}
			out.println("");
			
			int num_bytes = Integer.parseInt(parts[1]);
			if(parts.length != num_bytes + 2) {
				System.err.println("Error");
				System.exit(-1);
			}
		}


		out.println("\n\n\n\n\n\n");
		int num_pkt = 0;

		while(true) {
			String packet = c2.readLine();
			if(packet == null) {
				break;
			}

			String parts[] = packet.split(" ");
			if(!dfaMap.containsKey(parts[0])) {
				System.out.println("Error 2");
				System.exit(-1);
			}

			parts[0] = "" + dfaMap.get(parts[0]);
			for(int i = 0; i < parts.length; i ++) {
				out.print(parts[i] + " ");
			}

			out.println("");

			num_pkt ++;
			if(num_pkt % 10000 == 0) {
				System.err.println(num_pkt);
			}
		}

		out.close();
	}
}
