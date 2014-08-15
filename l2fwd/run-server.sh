# Re-building is required because the same executable does not
# work at both the client and the server
make clean
make

sudo rm -rf /mnt/huge/*
sudo ipcrm -M 1			# BASE_HT_LOG_SHM_KEY = 1
#sudo ./build/l2fwd -c 0xAA55 -n 4
sudo ./build/l2fwd -c 0xAA -n 4

# AA = lcores 1, 3, 5, 7
