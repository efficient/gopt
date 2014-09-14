#make

for num_pkts in `seq 4 16 1024`; do
	time=`taskset -c 0 ./mappedMemory $num_pkts | grep 5th | cut -d' ' -f 4`
	echo "$num_pkts $time"
	sleep .5
done

