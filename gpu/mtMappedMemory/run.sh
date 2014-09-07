make

for num_pkts in `seq 1 1 256`; do
	time=`taskset -c 0 ./mtMappedMemory | grep Full | cut -d' ' -f 4`
	echo "$num_pkts $time"
	sleep .5
done

