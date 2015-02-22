import java.io.*;
import java.util.*;
import java.util.Arrays;

/* Print the average and percentiles of URL lengths */
public class proc {
	public static void main(String args[]) throws IOException {
		BufferedReader c = new BufferedReader(new FileReader("fib_1010"));
		long totLen = 0;
		int numEntries = 0;

		System.err.println("Reading fib_1010..");

		while(true) {
			String url = c.readLine();
			if(url == null) {
				break;
			}

			numEntries ++;
			totLen += url.length();
		}

		System.err.printf("Average URL length = %.2f\n",
			(float) totLen / numEntries);

		int lenArr[] = new int[numEntries];
		c = new BufferedReader(new FileReader("fib_1010"));

		for(int i = 0; i < numEntries; i ++) {
			String url = c.readLine();
			lenArr[i] = url.length();
		}

		Arrays.sort(lenArr);

		for(int i = 10; i <= 90; i +=10) {
			System.err.println(i + "th percentile of length = " +
				lenArr[(i * numEntries) / 100]);
		}

		/*< Focus on the tail */
		for(int i = 90; i <= 99; i ++) {
			System.err.println(i + "th percentile of length = " +
				lenArr[(i * numEntries) / 100]);
		}

		for(float i = 99; i <= 99.9; i += .1) {
			System.err.printf("%.1f th percentile of length = %d\n",
				i, lenArr[(int) (i * numEntries) / 100]);
		}
	}
}
