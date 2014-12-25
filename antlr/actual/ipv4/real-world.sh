# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue "Removing hugepages"
shm-rm.sh 1>/dev/null 2>/dev/null

blue "Running real-world"
sudo taskset -c 0 ./real-world
