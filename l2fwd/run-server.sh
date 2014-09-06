# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue "Re-compiling master's CUDA code"
nvcc -O3 -o master util.c master.cu -lrt

blue "Re-compiling DPDK code"
make clean
make

blue "Removing DPDK's hugepages and shm keys 1 and 2"
sudo rm -rf /mnt/huge/*
sudo ipcrm -M 1			# WM_QUEUE_KEY
sudo ipcrm -M 2			# IPv4_CACHE_KEY

blue "Running gpu master and sleeping for 1 seconds"
sudo taskset -c 0 ./master -c 0x2 &
sleep 1

blue "Running workers"
#sudo ./build/l2fwd -c 0xAA55 -n 4
sudo ./build/l2fwd -c 0x2 -n 4

# AA = lcores 1, 3, 5, 7
