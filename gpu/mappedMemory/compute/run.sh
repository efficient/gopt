for num_pkts in 16 128 512 4096 16384; do
	cpu_time=`taskset -c 0 ./cpuCompute $num_pkts | grep "Average" | cut -d' ' -f 2`
	gpu_time=`taskset -c 0 ./mappedCompute $num_pkts | grep "5th" | cut -d' ' -f 4`
	speedup=`python -c "print $cpu_time / $gpu_time"`
	echo "$num_pkts cpuTime $cpu_time us gpuTime $gpu_time us gpu speedup $speedup"
	sleep .5
done

