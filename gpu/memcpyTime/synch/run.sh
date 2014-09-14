for num_pkts in `seq 4 16 1024`; do
	time=`taskset -c 0 ./memcpyTime $num_pkts | grep TOT | cut -d' ' -f 6`
	echo "$num_pkts $time"
	sleep .5
done
