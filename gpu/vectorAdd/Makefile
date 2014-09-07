all:
	nvcc -O3 -o vectorAdd vectorAdd.cu common.cu -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 vectorAdd.cu common.cu -lrt
clean:
	rm vectorAdd
