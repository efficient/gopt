# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

blue "Re-compiling master's CUDA code"
nvcc -O3  -o master master.cu -lrt

blue "Re-compiling DPDK code"
make
