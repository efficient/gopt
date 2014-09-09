# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

worker_core_mask="0x55"		# Mask for lcores running DPDK-workers
blue "Re-compiling DPDK code"
make clean
make

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
	echo "Usage: ./run_client.sh <0/1>"
	exit
fi

blue "Removing DPDK's hugepages and shm keys 1 and 2"
sudo rm -rf /mnt/huge/*
sudo ipcrm -M 1
sudo ipcrm -M 2

sudo ./build/l2fwd -c 0x15 -n 3 client $@		#AAA means all odd numbered cores

# Core masks: The assignment of lcores to ports is fixed. 
# 	int lcore_to_port[12] = {0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3};
# 0x15 = Lcores for xge0
# 0x555 = Lcores for xge0,1
# 0xAAA = Lcores for xge2,3
# 0x2A = Lcores for xge2
