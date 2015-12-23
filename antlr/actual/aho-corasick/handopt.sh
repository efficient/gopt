# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue "Removing hugepages for shm 0 to 450"
for i in `seq 0 450`; do
	sudo ipcrm -M $i 1>/dev/null 2>/dev/null
done

blue "Running handopt"
shm-rm.sh 1>/dev/null 2>/dev/null
sudo numactl --cpunodebind=0 --membind=0 ./handopt 28
