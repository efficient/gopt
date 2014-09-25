# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

sum=0
for i in `seq 0 1 127`; do
	new_prefixes=`cat uniq_ipv6_rib_201409 | grep "/$i" | wc -l`
	sum=`expr $sum + $new_prefixes`
	blue "New prefixes of length $i = $new_prefixes"
done
