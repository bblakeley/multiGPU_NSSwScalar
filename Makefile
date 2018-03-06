NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
LIBRARIES = -L/usr/local/cuda/lib -lcufft 
INCLUDES = -I/home/bblakeley/NVIDIA_CUDA-9.0_Samples/common/inc -I/usr/local/cuda/inc

main: multiGPU_3D_NSSwScalar.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(INCLUDES) $(LIBRARIES)

clean:
	rm -f *.o *.exe

