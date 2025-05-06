#ifndef LSM_CPU_H
#define LSM_CPU_H

#include <vector>

struct PathData {
    std::vector<float> prices;
    float cashflow;
    int remaining;
};

unsigned long long total_flops_lsm_cpu();
std::vector<PathData> simulate_paths_lsm_cpu();
bool solve_3x3_lsm_cpu(const float A[3][3], const float B[3], float X[3]);
extern "C" void lsm_cpu();

#endif
