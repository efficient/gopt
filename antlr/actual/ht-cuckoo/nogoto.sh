# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue ""
blue "Running handopt"
shm-rm.sh 1>/dev/null 2>/dev/null
num_threads=8

for i in `seq 1 $num_threads`; do
	sudo numactl --cpunodebind=0 --membind=0 ./nogoto $i &
done
