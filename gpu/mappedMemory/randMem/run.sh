# For fawn-haswell, don't go up to 16384
for num_pkts in 16 128 512 4096 16384; do
	shm-rm.sh 1>/dev/null 2>/dev/null
	gpu_out=`sudo taskset -c 0 ./mappedRandMem $num_pkts | grep "Cachelines"`
	echo $gpu_out
done

