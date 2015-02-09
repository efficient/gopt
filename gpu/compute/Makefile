all:
	nvcc -O3 -o compute compute.cu common.cu city.c -lrt
opt:
	nvcc -O3 -o compute -gencode arch=compute_35,code=compute_35 compute.cu common.cu city.c -lrt
clean:
	rm compute
