all:
	nvcc -O3 -o memcpyTime memcpyTime.cu common.cu -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 memcpyTime.cu common.cu -lrt
clean:
	rm memcpyTime
