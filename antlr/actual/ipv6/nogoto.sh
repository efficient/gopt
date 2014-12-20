# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

use_random_prefixes=1

blue "Removing hugepages"
shm-rm.sh 1>/dev/null 2>/dev/null

blue "Running nogoto with use_random_prefixes = $use_random_prefixes"
sudo taskset -c 0 ./nogoto $use_random_prefixes
