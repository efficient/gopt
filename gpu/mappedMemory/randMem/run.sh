for num_pkts in 16 128 512 4096 16384; do
	shm-rm.sh 1>/dev/null 2>/dev/null
	cpu_time=`sudo taskset -c 0 ./cpuRandMem $num_pkts | grep "Average" | cut -d' ' -f 2`
	gpu_time=`sudo taskset -c 0 ./mappedRandMem $num_pkts | grep "Average" | cut -d' ' -f 2`
	speedup=`python -c "print $cpu_time / $gpu_time"`
	echo "$num_pkts cpuTime $cpu_time us gpuTime $gpu_time us gpu speedup $speedup"
	sleep .5
done

