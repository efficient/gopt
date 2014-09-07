all:
	nvcc -O3 -o randMem randMem.cu common.cu -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 randMem.cu common.cu -lrt
clean:
	rm randMem
