NVCC = nvcc
COMMON_INC = -I/scratch/cuda-samples/Common
NVCCFLAGS = -arch=sm_70 -O3 -Xcompiler -fPIC

TARGET = driver
OBJS = lsm_gpu.o lsm_cpu.o qt2.o qt3.o

all: $(TARGET)

lsm_cpu.o: lsm_cpu.cu
	$(NVCC) -c $(NVCCFLAGS) lsm_cpu.cu -o lsm_cpu.o

lsm_gpu.o: lsm_gpu.cu
	$(NVCC) -c $(NVCCFLAGS) $(COMMON_INC) lsm_gpu.cu -o lsm_gpu.o -lcublas

qt2.o: qt2.cu
	$(NVCC) -c $(NVCCFLAGS) qt2.cu -o qt2.o -lcurand

qt3.o: qt3.cu
	$(NVCC) -c $(NVCCFLAGS) qt3.cu -o qt3.o -lcurand

$(TARGET): driver.cu $(OBJS)
	$(NVCC) $(NVCCFLAGS) driver.cu $(OBJS) -o $(TARGET) -lcublas -lcurand

clean:
	rm -f *.o $(TARGET)