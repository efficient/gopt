# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue ""
blue "Running nogoto"
shm-rm.sh 1>/dev/null 2>/dev/null
sudo taskset -c 0 ./nogoto

blue ""
blue "Running goto"
shm-rm.sh 1>/dev/null 2>/dev/null
sudo taskset -c 0 ./goto

blue ""
blue "Running handopt"
shm-rm.sh 1>/dev/null 2>/dev/null
sudo taskset -c 0 ./handopt
