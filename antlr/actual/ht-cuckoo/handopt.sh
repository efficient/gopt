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
sudo numactl --physcpubind=0-7 --membind=0 ./handopt $num_threads
