# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue "Recompiling cpu.c"
make cpu

blue "Removing existing shm regions"
shm-rm.sh 1>/dev/null 2>/dev/null

num_proc=8
blue "Running cpu.c on $num_proc cores"

for i in `seq 0 7`; do
	tid=`expr $i + 1`
	core=`expr $i \* 2`
	taskset -c $core ./cpu &
done
