all:
	nvcc -O3 -o mappedRandMem mappedRandMem.cu common.cu -lrt
	gcc -O3 -o cpuRandMem cpuRandMem.c -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 mappedRandMem.cu common.cu -lrt
clean:
	rm mappedRandMem cpuRandMem
