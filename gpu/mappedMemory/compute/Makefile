all:
	nvcc -O3 -o mappedCompute mappedCompute.cu common.cu city.c -lrt
	gcc -O3 -o cpuCompute cpuCompute.c city.c -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 mappedCompute.cu common.cu city.c -lrt
clean:
	rm mappedCompute cpuCompute
