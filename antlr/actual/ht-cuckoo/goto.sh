# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue ""
blue "Running handopt"
shm-rm.sh 1>/dev/null 2>/dev/null

# xia-r2
# cpus=0,2,4,6,8,10,12,14

# Apt c6220
# cpus=0-7

num_threads=16

# Use all hyperthreads on this socket
sudo numactl --cpunodebind=0 --membind=0 ./goto $num_threads
