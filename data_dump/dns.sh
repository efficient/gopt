# A function to echo in blue color
function blue() {
	es=`tput setaf 4`
	ee=`tput sgr0`
	echo "${es}$1${ee}"
}

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
	echo "Usage: ./dns.sh <index of input URL file in /proj/fawn/akalia/urls_inp/"
	exit
fi

input_file="/proj/fawn/akalia/urls_inp/urls_$1"
output_file="/proj/fawn/akalia/urls_out/urls_$1"

blue "Using input URL file $input_file"
blue "Using output URL file $output_file"

#Iterate over URLs in input file
while read new_url; do
	dns_out=`dig +short $new_url`
	if [[ -n $dns_out ]]; 
	then
		echo $new_url $dns_out >> $output_file
		echo "File $input_file: $new_url $dns_out"
	else
		echo "$new_url fail" >> $output_file
		echo "File $input_file: $new_url fail"
	fi
done < $input_file

