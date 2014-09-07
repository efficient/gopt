#!/bin/bash

CWD=`pwd`

echo "If figures don't appear in the pdf output, use make fig"

# If the command line argument is fig, make all figures
if [ $@ == "fig" ]
  then
  	echo "Making figures"
	for fig_dir in verbs-latency verbs-latency/diagram \
	verbs-throughput/sender verbs-throughput/sender/diagram \
	verbs-throughput/receiver verbs-throughput/receiver/diagram \
	ud-uc-comp/fanout \
	system-tput/small-items system-tput/var-size/ib system-tput/var-size/roce \
	system-tput/skew system-tput/scalability \
	system-latency \
	memory-depth \
	diagrams/verbs diagrams/index-types diagrams/request-region \
	echo cpu-usage; do
		echo "Building figures in directory figures/$fig_dir"
		cd figures/$fig_dir;./do.sh 1
		cd $CWD
	done

	make clean
	make pdf
fi

# If the command line argument is fig-clean, delete all figures
if [ $@ == "fig-clean" ]
  then
  	echo "Cleaning up figures"
	for fig_dir in verbs-latency verbs-latency/diagram \
	verbs-throughput/sender verbs-throughput/sender/diagram \
	verbs-throughput/receiver verbs-throughput/receiver/diagram \
	ud-uc-comp/fanout \
	system-tput/small-items system-tput/var-size/ib system-tput/var-size/roce \
	system-tput/skew system-tput/scalability \
	system-latency \
	memory-depth \
	diagrams/verbs diagrams/index-types diagrams/request-region \
	echo cpu-usage; do
		echo "Removing figures in directory figures/$fig_dir"
		cd figures/$fig_dir; rm *.aux *.pdf *.ps 2>/dev/null
		cd $CWD
	done
fi

make clean
make pdf
open paper.pdf
