# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

# Test if the IPv6 lookup kernel is working properly
blue "Re-compiling master's CUDA code"
nvcc -O3 -o master -gencode arch=compute_35,code=compute_35 util.c ipv6.c rte_lpm6.c master.cu -lrt -lnuma

blue "Killing existing GPU-master processes"
sudo killall master

worker_lcore_mask="0x1"
sudo taskset -c 14 ./master -c $worker_lcore_mask
