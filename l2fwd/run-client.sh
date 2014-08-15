# Re-building is required because the same executable does not
# work at both the client and the server
make clean
make 

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
	echo "Usage: ./run_client.sh <0/1>"
	exit
fi

sudo ./build/l2fwd -c 0xAAA -n 3 client $@		#AAA means all odd numbered cores

# Core masks: The assignment of lcores to ports is fixed. 
# 	int lcore_to_port[12] = {0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3};
# 0x15 = Lcores for xge0
# 0x555 = Lcores for xge0,1
# 0xAAA = Lcores for xge2,3
