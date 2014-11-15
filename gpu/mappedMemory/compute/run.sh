# For fawn-haswell, don't go up to 16384
for num_pkts in 16 128 512 4096; do
	shm-rm.sh 1>/dev/null 2>/dev/null
	gpu_out=`sudo taskset -c 0 ./mappedCompute $num_pkts | grep "hashes"`
	echo $gpu_out
done

