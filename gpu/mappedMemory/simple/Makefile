all:
	nvcc -O3 -o mappedMemory mappedMemory.cu common.cu -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 mappedMemory.cu common.cu -lrt
clean:
	rm mappedMemory
