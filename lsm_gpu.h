#ifndef LSM_GPU_H
#define LSM_GPU_H

unsigned long long total_flops_lsm_gpu();
void solve_least_squares_lsm_gpu(float *h_XtX, float *h_XtY, float *h_coeff);
extern "C" void lsm_gpu();

__global__ void simulate_paths_lsm_gpu(float *d_paths, int num_timesteps, int num_paths);
__global__ void compute_xty_lsm_gpu(float *d_paths, float *d_cashflow, float *d_XtX, float *d_XtY, 
                                    int timestep, int num_paths, int num_basis);
__global__ void update_cashflow_lsm_gpu(float *d_paths, float *d_cashflow, int *d_remaining, 
                                        float *d_coeff, int timestep, int num_paths);

#endif
