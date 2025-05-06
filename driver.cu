#include <iostream>
#include <cuda_runtime.h>
#include "lsm_cpu.h"
#include "lsm_gpu.h"
#include "qt2.h"
#include "qt3.h"

void benchmark(void (*func)(), const std::string& name, float flops) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float gflops = (flops / 1e9f) / (ms / 1000.0f);

    std::cout << name << ": " << ms << " ms, " << gflops << " GFLOP/s\n";
}

int main() {
    benchmark(lsm_cpu, "lsm_cpu", 0);
    // benchmark(lsm_gpu, "lsm_gpu", 0);
    benchmark(qt2, "qt2", 0);
    benchmark(qt3, "qt3", 0);
    return 0;
}
