#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "lsm_gpu.h"

#define NUM_PATHS 100000    // Monte Carlo paths
#define NUM_TIMESTEPS 365
#define NUM_BASIS 3         // basis functions (1, S, S^2)
#define STRIKE 100.0f
#define RISKFREE 0.05f      // risk-free rate
#define VOLATILITY 0.2f
#define S0 100.0f           // initial asset price
#define MAX_EXERCISE 5
#define BLOCK_SIZE 256

// GBM
__global__ void simulate_paths_lsm_gpu(float *d_paths, int num_timesteps, int num_paths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    curandState state;
    curand_init(42, idx, 0, &state);
    float dt = 1.0f / num_timesteps;
    float s = S0;

    for (int t = 0; t < num_timesteps; ++t) {
        float dW = curand_normal(&state) * sqrtf(dt);
        s *= expf((RISKFREE - 0.5f * VOLATILITY * VOLATILITY) * dt + VOLATILITY * dW);
        d_paths[t * num_paths + idx] = s;
    }
}

// matrices for regression
__global__ void compute_xty_lsm_gpu(float *d_paths, float *d_cashflow, float *d_XtX, float *d_XtY, 
                            int timestep, int num_paths, int num_basis) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    float S = d_paths[timestep * num_paths + idx];
    float payoff = fmaxf(S - STRIKE, 0.0f);

    if (payoff > 1e-5) {  // in-the-money
        float X[NUM_BASIS] = {1.0f, S, S*S};
        float Y = d_cashflow[idx] * expf(-RISKFREE * (1.0f/NUM_TIMESTEPS));

        for (int i = 0; i < num_basis; ++i) {
            atomicAdd(&d_XtY[i], X[i] * Y);
            for (int j = 0; j < num_basis; ++j) {
                atomicAdd(&d_XtX[i * num_basis + j], X[i] * X[j]);
            }
        }
    }
}

// cashflows and exercise decisions
__global__ void update_cashflow_lsm_gpu(float *d_paths, float *d_cashflow, int *d_remaining, 
                                float *d_coeff, int timestep, int num_paths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    float S = d_paths[timestep * num_paths + idx];
    float immediate = fmaxf(S - STRIKE, 0.0f);
    float continuation = d_coeff[0] + d_coeff[1]*S + d_coeff[2]*S*S;

    if (immediate > continuation && d_remaining[idx] > 0) {
        d_cashflow[idx] = immediate;
        atomicSub(&d_remaining[idx], 1);
    } else {
        d_cashflow[idx] *= expf(-RISKFREE * (1.0f/NUM_TIMESTEPS));
    }
}

// solve linear system using cuBLAS
void solve_least_squares_lsm_gpu(float *h_XtX, float *h_XtY, float *h_coeff) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_XtX, *d_XtY;
    cudaMalloc(&d_XtX, NUM_BASIS * NUM_BASIS * sizeof(float));
    cudaMalloc(&d_XtY, NUM_BASIS * sizeof(float));
    
    cudaMemcpy(d_XtX, h_XtX, NUM_BASIS * NUM_BASIS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_XtY, h_XtY, NUM_BASIS * sizeof(float), cudaMemcpyHostToDevice);

    int info;
    int *d_info;
    cudaMalloc(&d_info, sizeof(int));
    cublasSgetrfBatched(handle, NUM_BASIS, &d_XtX, NUM_BASIS, NULL, &info, 1);
    cublasSgetrsBatched(handle, CUBLAS_OP_N, NUM_BASIS, 1, &d_XtX, NUM_BASIS, 
                        NULL, &d_XtY, NUM_BASIS, &info, 1);

    cudaMemcpy(h_coeff, d_XtY, NUM_BASIS * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_XtX);
    cudaFree(d_XtY);
}

extern "C" void lsm_gpu() {
    // allocate device
    float *d_paths, *d_cashflow;
    int *d_remaining;
    cudaMalloc(&d_paths, NUM_TIMESTEPS * NUM_PATHS * sizeof(float));
    cudaMalloc(&d_cashflow, NUM_PATHS * sizeof(float));
    cudaMalloc(&d_remaining, NUM_PATHS * sizeof(int));

    // init remaining exercises
    int h_remaining[NUM_PATHS];
    for (int i = 0; i < NUM_PATHS; ++i) h_remaining[i] = MAX_EXERCISE;
    cudaMemcpy(d_remaining, h_remaining, NUM_PATHS * sizeof(int), cudaMemcpyHostToDevice);

    simulate_paths_lsm_gpu<<<(NUM_PATHS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>
        (d_paths, NUM_TIMESTEPS, NUM_PATHS);

    // init final cashflows
    float *h_final_prices = new float[NUM_PATHS];
    cudaMemcpy(h_final_prices, d_paths + (NUM_TIMESTEPS-1)*NUM_PATHS, 
               NUM_PATHS * sizeof(float), cudaMemcpyDeviceToHost);
    
    float *h_cashflow = new float[NUM_PATHS];
    for (int i = 0; i < NUM_PATHS; ++i)
        h_cashflow[i] = fmaxf(h_final_prices[i] - STRIKE, 0.0f);
    cudaMemcpy(d_cashflow, h_cashflow, NUM_PATHS * sizeof(float), cudaMemcpyHostToDevice);

    // backward induction
    for (int t = NUM_TIMESTEPS-2; t >= 0; --t) {
        float *d_XtX, *d_XtY;
        cudaMalloc(&d_XtX, NUM_BASIS * NUM_BASIS * sizeof(float));
        cudaMalloc(&d_XtY, NUM_BASIS * sizeof(float));
        cudaMemset(d_XtX, 0, NUM_BASIS * NUM_BASIS * sizeof(float));
        cudaMemset(d_XtY, 0, NUM_BASIS * sizeof(float));

        // find X^T X and X^T Y
        compute_xty_lsm_gpu<<<(NUM_PATHS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>
            (d_paths, d_cashflow, d_XtX, d_XtY, t, NUM_PATHS, NUM_BASIS);

        // regression coefficients
        float h_XtX[NUM_BASIS * NUM_BASIS] = {0};
        float h_XtY[NUM_BASIS] = {0};
        cudaMemcpy(h_XtX, d_XtX, NUM_BASIS * NUM_BASIS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_XtY, d_XtY, NUM_BASIS * sizeof(float), cudaMemcpyDeviceToHost);

        float h_coeff[NUM_BASIS] = {0};
        solve_least_squares_lsm_gpu(h_XtX, h_XtY, h_coeff);

        // cashflows
        float *d_coeff;
        cudaMalloc(&d_coeff, NUM_BASIS * sizeof(float));
        cudaMemcpy(d_coeff, h_coeff, NUM_BASIS * sizeof(float), cudaMemcpyHostToDevice);

        update_cashflow_lsm_gpu<<<(NUM_PATHS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>
            (d_paths, d_cashflow, d_remaining, d_coeff, t, NUM_PATHS);

        cudaFree(d_XtX);
        cudaFree(d_XtY);
        cudaFree(d_coeff);
    }

    // final price
    cudaMemcpy(h_cashflow, d_cashflow, NUM_PATHS * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for (int i = 0; i < NUM_PATHS; ++i) sum += h_cashflow[i];
    float price = sum / NUM_PATHS;

    printf("Estimated Swing Option Price: %f\n", price);

    delete[] h_final_prices;
    delete[] h_cashflow;
    cudaFree(d_paths);
    cudaFree(d_cashflow);
    cudaFree(d_remaining);
}