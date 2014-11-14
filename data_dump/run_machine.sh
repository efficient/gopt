# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
	echo "Usage: ./run_machine.sh <machine_id>"
	exit
fi

machine_id=$1
for i in `seq 0 19`; do
	file_id=`expr $machine_id \* 20 + $i`
	if [ $file_id -lt 195 ]
	then
		./dns.sh $file_id &
	fi
done
