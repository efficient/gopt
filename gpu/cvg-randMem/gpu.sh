# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue "Removing hugepage key LOG_KEY"
sudo ipcrm -M 1

blue "Re-compiline gpu code"
make clean
make gpu

blue "Running GPU random access measurement"
sudo taskset -c 0 ./gpu
