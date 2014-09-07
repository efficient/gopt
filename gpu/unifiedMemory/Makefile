all:
	nvcc -O3 -o unifiedMemory unifiedMemory.cu common.cu -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 unifiedMemory.cu common.cu -lrt
clean:
	rm unifiedMemory
