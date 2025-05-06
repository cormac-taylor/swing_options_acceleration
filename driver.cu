#include <iostream>
#include <cuda_runtime.h>
#include "lsm_cpu.h"
#include "lsm_gpu.h"
#include "qt2.h"
#include "qt3.h"

void benchmark(void (*func)(), const std::string& name, unsigned long long flops) {
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
    benchmark(lsm_cpu, "lsm_cpu", total_flops_lsm_cpu());
    benchmark(lsm_gpu, "lsm_gpu", total_flops_lsm_gpu());
    benchmark(qt2, "qt2", total_flops_qt2());
    benchmark(qt3, "qt3", total_flops_qt3());
    return 0;
}
