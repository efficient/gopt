all:
	nvcc -O3 -o mtMappedMemory mtMappedMemory.cu common.cu -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 mtMappedMemory.cu common.cu -lrt
clean:
	rm mtMappedMemory
