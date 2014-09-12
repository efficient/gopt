all:
	nvcc -O3 -o seqMem seqMem.cu common.cu -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 seqMem.cu common.cu -lrt
clean:
	rm seqMem
