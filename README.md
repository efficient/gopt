##Contents:

* **branches**: The CPU and GPU code for 4 packet processing applications: IPv4 forwarding, IPv6 forwarding, L2 switching, and NDN forwarding is available in different branches of this repository.

* **antlr**: ANTLR code for the G-Opt transformation.

 * **antlr/actual**: Sample applications for benchmarking the transformation.
 * **antlr/actual/aho-corasick**: Optimized code for pattern matching in intrusion detection.

* **l2fwd**: DPDK code for full-system benchmarks (contents vary for different branches).

* **data_dump**: Contains data files for:
 1. Unique IPv4 and IPv6 prefixes from RouteViews.
 2. Processed IPv6 prefixes: suitable for trie insertion.
 3. Snort 2.9.7's valid rules, inserted patterns, and probed packets.

##Dependencies and limitations:
* This code has been tested with `DPDK 1.5.0r0` and `CUDA 6.0`.
* Some of the packet forwarding (at the server) and packet generation (at the clients) is specialized to our hardware. It should be straightforward to modify it for your hardware.


##License

	Copyright 2015 Carnegie Mellon University

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

	    http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
