#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define NUM_TIMESTEPS 365
#define NUM_PATHS 100000       // Monte Carlo paths
#define NUM_GRID 900           // grid size per timestep
#define BLOCK_SIZE 256
#define ALPHA 0.1f             // AR(1) coefficient
#define SIGMA 0.2f             // volatility
#define INITIAL_X 0.0f
#define STRIKE 100.0f
#define RISKFREE 0.05f         // risk-free rate
#define MAX_EXERCISE 5

// generate quantization grid
void generate_grid(float *grid, int size, float mean, float stddev) {
    for(int i=0; i<size; i++) {
        float u = (i + 0.5f) / size;
        grid[i] = mean + stddev * sqrtf(-2.0f * logf(u)) * cosf(2.0f * M_PI * u);
    }
}

// GBM
float grid_to_price(float x, float t) {
    return STRIKE * expf((RISKFREE - 0.5f*SIGMA*SIGMA)*t + SIGMA*x);
}

// compute transitions and track remaining exercises
__global__ void compute_transitions(int *d_p_ij, int *d_p_i, float *d_grid, 
                                   int num_timesteps, int num_paths, int grid_size) {
    int timestep = blockIdx.x;
    int path_idx = threadIdx.x + blockIdx.y * blockDim.x;
    if (timestep >= num_timesteps || path_idx >= num_paths) return;

    __shared__ float s_current_grid[NUM_GRID];
    __shared__ float s_next_grid[NUM_GRID];
    if (threadIdx.x < NUM_GRID) {
        s_current_grid[threadIdx.x] = d_grid[timestep * NUM_GRID + threadIdx.x];
        if (timestep < num_timesteps-1)
            s_next_grid[threadIdx.x] = d_grid[(timestep+1) * NUM_GRID + threadIdx.x];
    }
    __syncthreads();

    curandState state;
    curand_init(1234, path_idx, 0, &state);

    float X_prev = INITIAL_X;
    for(int k=0; k < timestep; k++) {
        float epsilon = curand_normal(&state);
        X_prev = ALPHA * X_prev + SIGMA * epsilon;
    }

    float epsilon = curand_normal(&state);
    float X_next = ALPHA * X_prev + SIGMA * epsilon;

    int i_min = 0, j_min = 0;
    float dist_i = INFINITY, dist_j = INFINITY;
    for(int idx=0; idx<NUM_GRID; idx++) {
        float d_current = fabsf(X_prev - s_current_grid[idx]);
        float d_next = fabsf(X_next - s_next_grid[idx]);
        if(d_current < dist_i) { dist_i = d_current; i_min = idx; }
        if(d_next < dist_j) { dist_j = d_next; j_min = idx; }
    }

    atomicAdd(&d_p_ij[timestep * NUM_GRID * NUM_GRID + i_min * NUM_GRID + j_min], 1);
    atomicAdd(&d_p_i[timestep * NUM_GRID + i_min], 1);
}

// backward induction on host
void backward_induction(float *grid, int *p_ij, int *p_i, float *values) {
    float dt = 1.0f / NUM_TIMESTEPS;
    
    // init final timestep
    for(int i=0; i<NUM_GRID; i++) {
        float S_T = grid_to_price(grid[(NUM_TIMESTEPS-1)*NUM_GRID + i], 1.0f);
        values[(NUM_TIMESTEPS-1)*NUM_GRID + i] = fmaxf(S_T - STRIKE, 0.0f);
    }

    // backward iteration
    for(int k=NUM_TIMESTEPS-2; k>=0; k--) {
        for(int i=0; i<NUM_GRID; i++) {
            float immediate = fmaxf(grid_to_price(grid[k*NUM_GRID + i], k*dt) - STRIKE, 0.0f);
            
            float continuation = 0.0f;
            int total = p_i[k*NUM_GRID + i];
            if(total > 0) {
                for(int j=0; j<NUM_GRID; j++) {
                    float prob = (float)p_ij[k*NUM_GRID*NUM_GRID + i*NUM_GRID + j] / total;
                    continuation += prob * values[(k+1)*NUM_GRID + j];
                }
                continuation *= expf(-RISKFREE * dt);
            }
            
            values[k*NUM_GRID + i] = fmaxf(immediate, continuation);
        }
    }
}

int main() {
    float *h_grid = new float[NUM_TIMESTEPS * NUM_GRID];
    float mean = 0.0f;
    float stddev = SIGMA / sqrtf(1 - ALPHA*ALPHA);
    for(int k=0; k<NUM_TIMESTEPS; k++)
        generate_grid(h_grid + k*NUM_GRID, NUM_GRID, mean, stddev);

    float *d_grid;
    int *d_p_ij, *d_p_i;
    cudaMalloc(&d_grid, NUM_TIMESTEPS * NUM_GRID * sizeof(float));
    cudaMemcpy(d_grid, h_grid, NUM_TIMESTEPS * NUM_GRID * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_p_ij, NUM_TIMESTEPS * NUM_GRID * NUM_GRID * sizeof(int));
    cudaMalloc(&d_p_i, NUM_TIMESTEPS * NUM_GRID * sizeof(int));
    cudaMemset(d_p_ij, 0, NUM_TIMESTEPS * NUM_GRID * NUM_GRID * sizeof(int));
    cudaMemset(d_p_i, 0, NUM_TIMESTEPS * NUM_GRID * sizeof(int));

    dim3 grid_dim(NUM_TIMESTEPS, (NUM_PATHS + BLOCK_SIZE-1)/BLOCK_SIZE);
    compute_transitions<<<grid_dim, BLOCK_SIZE>>>(d_p_ij, d_p_i, d_grid, 
                                                 NUM_TIMESTEPS, NUM_PATHS, NUM_GRID);

    int *h_p_ij = new int[NUM_TIMESTEPS * NUM_GRID * NUM_GRID];
    int *h_p_i = new int[NUM_TIMESTEPS * NUM_GRID];
    cudaMemcpy(h_p_ij, d_p_ij, NUM_TIMESTEPS * NUM_GRID * NUM_GRID * sizeof(int), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p_i, d_p_i, NUM_TIMESTEPS * NUM_GRID * sizeof(int), 
               cudaMemcpyDeviceToHost);

    float *h_values = new float[NUM_TIMESTEPS * NUM_GRID];
    backward_induction(h_grid, h_p_ij, h_p_i, h_values);

    float price = 0.0f;
    for(int i=0; i<NUM_GRID; i++) 
        price += h_values[i];
    price /= NUM_GRID;
    printf("Estimated Swing Option Price: %.4f\n", price);

    delete[] h_grid; delete[] h_p_ij; delete[] h_p_i; delete[] h_values;
    cudaFree(d_grid); cudaFree(d_p_ij); cudaFree(d_p_i);
    return 0;
}