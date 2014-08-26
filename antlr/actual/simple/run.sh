for compute in `seq 1 5 50`; do
	echo "#define COMPUTE $compute" > param.h
	make 1>/dev/null 2>/dev/null

	echo "COMPUTE = $compute, nogoto:"
	sudo numactl --physcpubind 0 --interleave 0 ./nogoto
	shm-rm.sh 1>/dev/null 2>/dev/null
	echo ""

	echo "COMPUTE = $compute, goto:"
	sudo numactl --physcpubind 0 --interleave 0 ./goto
	shm-rm.sh 1>/dev/null 2>/dev/null
	echo ""

	echo "COMPUTE = $compute, manual:"
	sudo numactl --physcpubind 0 --interleave 0 ./manual
	shm-rm.sh 1>/dev/null 2>/dev/null
	echo ""
done
