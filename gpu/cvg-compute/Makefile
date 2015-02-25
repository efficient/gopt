gpu:
	nvcc -O3 -o gpu -gencode arch=compute_35,code=compute_35 gpu.cu common.c city.c -lrt
clean:
	rm gpu
