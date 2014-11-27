# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue ""
blue "Running handopt"
shm-rm.sh 1>/dev/null 2>/dev/null
num_threads=8

# xia-r2
cpus=0,2,4,6,8,10,12,14,16

# Apt c6220
cpus=0-7

for i in `seq 1 $num_threads`; do
	sudo numactl --physcpubind=$cpus --membind=0 ./goto $i &
done
