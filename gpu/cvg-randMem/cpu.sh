num_threads=1
sudo taskset -c 0,2,4,6,8,10,12,14 ./cpu $num_threads
