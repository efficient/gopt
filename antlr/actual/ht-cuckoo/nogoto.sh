# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue ""
blue "Running handopt"
shm-rm.sh 1>/dev/null 2>/dev/null

# Hyperthreading: use all hardware threads on socket #0
num_threads=16
sudo numactl --cpunodebind=0 --membind=0 ./nogoto $num_threads
