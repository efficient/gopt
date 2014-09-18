# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

worker_core_mask="0x1"		# Mask for lcores running DPDK-workers

blue "Re-compiling DPDK code"
make clean
make

blue "Removing DPDK's hugepages"
sudo rm -rf /mnt/huge/*

blue "Running workers"
sudo ./build/l2fwd -c $worker_core_mask -n 4

# AA = lcores 1, 3, 5, 7
# 55 = lcores 0, 2, 4, 6
