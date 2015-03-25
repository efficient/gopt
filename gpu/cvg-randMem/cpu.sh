# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue "Removing hugepage key LOG_KEY"
sudo ipcrm -M 1

blue "Re-compiling cpu code"
make clean
make cpu

num_threads=8
sudo taskset -c 0,2,4,6,8,10,12,14 ./cpu $num_threads

# 2697 v3
# num_threads=14
# sudo taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13 ./cpu $num_threads
