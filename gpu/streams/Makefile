all:
	nvcc -O3 -o streams streams.cu common.cu -lrt
opt:
	nvcc -O3 -gencode arch=compute_35,code=compute_35 streams.cu common.cu -lrt
clean:
	rm streams
