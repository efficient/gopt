all:
	nvcc -O3 -o compute compute.cu common.cu -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 compute.cu common.cu -lrt
clean:
	rm compute
