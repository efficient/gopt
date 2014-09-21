# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

worker_core_mask="0x55"		# Mask for lcores running DPDK-workers

blue "Killing existing GPU-master processes"
sudo killall master

blue "Re-compiling master's CUDA code"
nvcc -O3 -o master -gencode arch=compute_30,code=compute_30 util.c city.c cuckoo.c master.cu -lrt

blue "Re-compiling DPDK code"
make clean
make

blue "Removing DPDK's hugepages and shm keys 1 and 2"
sudo rm -rf /mnt/huge/*
sudo ipcm -M 1			# WM_QUEUE_KEY
sudo ipcrm -M 2			# HASH_INDEX_KEY

blue "Running gpu master on core 15 and sleeping for 2 seconds"
sudo taskset -c 14 ./master -c $worker_core_mask &
sleep 2

blue "Running workers"
sudo ./build/l2fwd -c $worker_core_mask -n 4

# AA = lcores 1, 3, 5, 7
# 55 = lcores 0, 2, 4, 6
