# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

echo ""
blue "Nogoto:"
sudo numactl --physcpubind 0 --interleave 0 ./nogoto

echo ""
blue "Goto:"
sudo numactl --physcpubind 0 --interleave 0 ./goto

echo ""
blue "Handopt:"
sudo numactl --physcpubind 0 --interleave 0 ./handopt

