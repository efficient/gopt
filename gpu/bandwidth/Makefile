all:
	nvcc -O3 -o bandwidthtest bandwidthtest.cu -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 bandwidthtest.cu common.cu -lrt
clean:
	rm bandwidthtest
