for compute in `seq 1 5 50`; do
	echo "#define COMPUTE $compute" > param.h
	make

	echo "COMPUTE = $compute, nogoto:"
	nogoto_result=`sudo ./nogoto 2>/dev/null`
	nogoto_time=`echo $nogoto_result | cut -d' ' -f 3`
	shm-rm.sh 1>/dev/null 2>/dev/null

	echo "COMPUTE = $compute, goto:"
	goto_result=`sudo ./goto 2>/dev/null`
	goto_time=`echo $goto_result | cut -d' ' -f 3`
	shm-rm.sh 1>/dev/null 2>/dev/null

	echo "COMPUTE = $compute, manual:"
	manual_result=`sudo ./manual goto 2>/dev/null`
	manual_time=`echo $manual_result | cut -d' ' -f 3`
	shm-rm.sh 1>/dev/null 2>/dev/null

	echo "nogoto_time = $nogoto_time, goto_time = $goto_time, manual_time = $manual_time"
	
done
