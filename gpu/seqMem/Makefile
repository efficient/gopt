all:
	nvcc -O3 -o seqMem seqMem.cu common.cu -lrt
opt:
	nvcc -O3 -o seqMem -gencode arch=compute_35,code=compute_35 seqMem.cu common.cu -lrt
d2d:
	nvcc -O3 -o d2dMemcpy -gencode arch=compute_35,code=compute_35 d2dMemcpy.cu common.cu -lrt
clean:
	rm seqMem cpu d2dMemcpy
cpu: cpu.c
	gcc -O3 -o cpu cpu.c -lrt
