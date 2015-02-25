gpu:
	nvcc -O3 -o gpu -gencode arch=compute_35,code=compute_35 gpu.cu common.c city.c -lrt
cpu: cpu.c common.c common.h
	gcc -O3 -o cpu cpu.c common.c city.c -lrt -lpthread
clean:
	rm gpu cpu
