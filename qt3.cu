#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "qt3.h"

#define NUM_TIMESTEPS 365
#define NUM_PATHS 100000       // Monte Carlo paths
#define NUM_GRID 500           // grid size per timestep
#define BLOCK_SIZE 256
#define ALPHA 0.1f             // AR(1) coefficient
#define SIGMA 0.2f             // volatility
#define STRIKE 100.0f
#define RISKFREE 0.05f         // risk-free rate

unsigned long long total_flops_qt3(){
    return (1LL * NUM_PATHS * NUM_TIMESTEPS * (NUM_TIMESTEPS + 4 * NUM_GRID)) +
            (8LL * NUM_GRID + 1LL * NUM_TIMESTEPS * NUM_GRID * 2 * NUM_GRID);
}

// generate quantization grid
void generate_grid_qt3(float *grid, int size, float mean, float stddev) {
    for(int i=0; i<size; i++) {
        float u = (i + 0.5f) / size;
        grid[i] = mean + stddev * sqrtf(-2.0f * logf(u)) * cosf(2.0f * M_PI * u);
    }
}

// compute transitions
__global__ void compute_transitions_qt3(int *d_p_ij, int *d_p_i, float *d_grid, 
                                   int num_timesteps, int num_paths, int grid_size,
                                   float stddev) {
    int k = blockIdx.x;
    int path_idx = threadIdx.x + blockIdx.y * blockDim.x;
    if (k >= num_timesteps || path_idx >= num_paths) return;

    __shared__ float s_grid_current[NUM_GRID];
    __shared__ float s_grid_next[NUM_GRID];
    if (threadIdx.x < NUM_GRID) {
        s_grid_current[threadIdx.x] = d_grid[k * NUM_GRID + threadIdx.x];
        if (k < num_timesteps - 1)
            s_grid_next[threadIdx.x] = d_grid[(k+1) * NUM_GRID + threadIdx.x];
    }
    __syncthreads();

    curandState state;
    curand_init(1234 + k, path_idx, 0, &state);

    float X_k = curand_normal(&state) * stddev;

    float epsilon = curand_normal(&state);

    float X_kp1 = ALPHA * X_k + SIGMA * epsilon;

    int i_min = 0, j_min = 0;
    float dist_i = INFINITY, dist_j = INFINITY;
    for(int idx=0; idx<NUM_GRID; idx++) {
        // search current grid
        float d_current = fabsf(X_k - s_grid_current[idx]);
        if(d_current < dist_i) {
            dist_i = d_current;
            i_min = idx;
        }
        
        // search next grid
        if(k < num_timesteps - 1) {
            float d_next = fabsf(X_kp1 - s_grid_next[idx]);
            if(d_next < dist_j) {
                dist_j = d_next;
                j_min = idx;
            }
        }
    }

    if(k < num_timesteps - 1) {
        atomicAdd(&d_p_ij[k * NUM_GRID * NUM_GRID + i_min * NUM_GRID + j_min], 1);
        atomicAdd(&d_p_i[k * NUM_GRID + i_min], 1);
    }
}

// backward induction
void backward_induction_qt3(float *grid, int *p_ij, int *p_i, float *values) {
    float dt = 1.0f / NUM_TIMESTEPS;
    
    for(int i=0; i<NUM_GRID; i++) {
        float S_T = STRIKE * expf((RISKFREE - 0.5f*SIGMA*SIGMA)*1.0f + SIGMA*grid[(NUM_TIMESTEPS-1)*NUM_GRID + i]);
        values[(NUM_TIMESTEPS-1)*NUM_GRID + i] = fmaxf(S_T - STRIKE, 0.0f);
    }

    for(int k=NUM_TIMESTEPS-2; k>=0; k--) {
        for(int i=0; i<NUM_GRID; i++) {
            float S_t = STRIKE * expf((RISKFREE - 0.5f*SIGMA*SIGMA)*(k*dt) + SIGMA*grid[k*NUM_GRID + i]);
            float immediate = fmaxf(S_t - STRIKE, 0.0f);
            
            float continuation = 0.0f;
            int total = p_i[k*NUM_GRID + i];
            if(total > 0) {
                for(int j=0; j<NUM_GRID; j++) {
                    int count = p_ij[k*NUM_GRID*NUM_GRID + i*NUM_GRID + j];
                    if(count > 0) {
                        float prob = (float)count / total;
                        continuation += prob * values[(k+1)*NUM_GRID + j];
                    }
                }
                continuation *= expf(-RISKFREE * dt);
            }
            
            values[k*NUM_GRID + i] = fmaxf(immediate, continuation);
        }
    }
}

extern "C" void qt3() {
    float stddev = SIGMA / sqrtf(1.0f - ALPHA*ALPHA);
    
    float *h_grid = new float[NUM_TIMESTEPS * NUM_GRID];
    for(int k=0; k<NUM_TIMESTEPS; k++)
        generate_grid_qt3(h_grid + k*NUM_GRID, NUM_GRID, 0.0f, stddev);

    float *d_grid;
    int *d_p_ij, *d_p_i;
    cudaMalloc(&d_grid, NUM_TIMESTEPS * NUM_GRID * sizeof(float));
    cudaMemcpy(d_grid, h_grid, NUM_TIMESTEPS * NUM_GRID * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_p_ij, NUM_TIMESTEPS * NUM_GRID * NUM_GRID * sizeof(int));
    cudaMalloc(&d_p_i, NUM_TIMESTEPS * NUM_GRID * sizeof(int));
    cudaMemset(d_p_ij, 0, NUM_TIMESTEPS * NUM_GRID * NUM_GRID * sizeof(int));
    cudaMemset(d_p_i, 0, NUM_TIMESTEPS * NUM_GRID * sizeof(int));

    dim3 grid_dim(NUM_TIMESTEPS, (NUM_PATHS + BLOCK_SIZE-1)/BLOCK_SIZE);
    compute_transitions_qt3<<<grid_dim, BLOCK_SIZE>>>(d_p_ij, d_p_i, d_grid, 
                                                 NUM_TIMESTEPS, NUM_PATHS, NUM_GRID,
                                                 stddev);

    int *h_p_ij = new int[NUM_TIMESTEPS * NUM_GRID * NUM_GRID];
    int *h_p_i = new int[NUM_TIMESTEPS * NUM_GRID];
    cudaMemcpy(h_p_ij, d_p_ij, NUM_TIMESTEPS * NUM_GRID * NUM_GRID * sizeof(int), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p_i, d_p_i, NUM_TIMESTEPS * NUM_GRID * sizeof(int), 
               cudaMemcpyDeviceToHost);

    float *h_values = new float[NUM_TIMESTEPS * NUM_GRID];
    backward_induction_qt3(h_grid, h_p_ij, h_p_i, h_values);

    float price = 0.0f;
    for(int i=0; i<NUM_GRID; i++) 
        price += h_values[i];
    price /= NUM_GRID;
    printf("Estimated Swing Option Price: %.4f\n", price);

    delete[] h_grid; delete[] h_p_ij; delete[] h_p_i; delete[] h_values;
    cudaFree(d_grid); cudaFree(d_p_ij); cudaFree(d_p_i);
}