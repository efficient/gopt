# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

rm master

# Test if the IPv6 lookup kernel is working properly
blue "Re-compiling master's CUDA code"
nvcc -O3 -o master -gencode arch=compute_35,code=compute_35 util.c city.c ndn.c master.cu -lrt

blue "Killing existing GPU-master processes"
sudo killall master

blue "Removing DPDK's hugepages and shm key 1, 2, 3"
sudo rm -rf /mnt/huge/*
sudo ipcrm -M 1			# WM_QUEUE_KEY
sudo ipcrm -M 2			# NDN_HT_INDEX_KEY
sudo ipcrm -M 3			# NDN_NAMES_KEY

worker_lcore_mask="0x1"
sudo taskset -c 14 ./master -c $worker_lcore_mask
