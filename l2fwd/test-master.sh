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

blue "Removing DPDK's hugepages and shm key 1, 2, 3"
sudo rm -rf /mnt/huge/*
sudo ipcrm -M 1			# WM_QUEUE_KEY
sudo ipcrm -M 2			# RTE_LPM6_SHM_KEY

worker_lcore_mask="0x1"
sudo taskset -c 14 ./master -c $worker_lcore_mask
