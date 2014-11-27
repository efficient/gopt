# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue ""
blue "Running handopt"
shm-rm.sh 1>/dev/null 2>/dev/null
num_threads=1

for i in `seq 1 $num_threads`; do
	sudo numactl --physcpubind=0,1,2,3 --membind=0 ./handopt $i &
done
