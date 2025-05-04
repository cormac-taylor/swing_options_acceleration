#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define CALL        true
#define PUT         false
#define MAX_STEPS   366
#define MAX_DIM     2
#define MAX_GRID    500
#define MAX_BASIS   5
#define DISCOUNT(dt, r) (expf(-(r) * (dt)))

typedef struct {
    // Simulation control
    int M_PATHS;       // num Monte Carlo paths
    int N_STEPS;       // num time steps
    float T;           // years to maturity
    float dT;          // time step size

    // Option parameters
    float S0;          // initial asset price
    float K;           // strike price
    float r;           // risk-free rate
    int Q_min, Q_max;  // min and max exercise rights

    // Longstaff-Schwartz
    int N_BASIS;       // num basis functions in regression

    // Quantization
    int N_GRID;        // num grid points per time step
    int N_DIM;         // state space dim: 1 for GBM, 2 for OU

    unsigned long RNG_SEED;

    // AR(1) 
    float alpha1;
    float alpha2;
    float sigma1;
    float sigma2;
} SimulationParams;

typedef struct {
    float X0[MAX_DIM];                          // initial state
    float A[MAX_STEPS][MAX_DIM][MAX_DIM];       // A_k transition matrices
    float Tmat[MAX_STEPS][MAX_DIM][MAX_DIM];    // T_k volatility matrices
} AR1ModelParams;

__global__ void init_rng_kernel(curandState *states, unsigned long seed, int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M)
        curand_init(seed, tid, 0, &states[tid]);
}

__global__ void simulate_paths_kernel(
    float *d_paths,
    const AR1ModelParams *d_ar1,
    curandState *d_rng_states,
    const int M_PATHS,
    const int N_STEPS
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M_PATHS) return;

    const int N_DIM = 2;
    float X_curr[N_DIM];
    float X_next[N_DIM];

    // load initial state
    for (int d = 0; d < N_DIM; d++) {
        X_curr[d] = d_ar1->X0[d];
        d_paths[(d * N_STEPS + 0) * M_PATHS + tid] = X_curr[d];
    }

    curandState local_state = d_rng_states[tid];

    for (int k = 0; k < N_STEPS - 1; k++) {
        float Z[N_DIM];
        for (int d = 0; d < N_DIM; d++) {
            Z[d] = curand_normal(&local_state);
        }

        // X_{k+1} = A_k * X_k + T_k * Z
        for (int i = 0; i < N_DIM; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N_DIM; j++) {
                sum += d_ar1->A[k][i][j] * X_curr[j];
            }
            for (int j = 0; j < N_DIM; j++) {
                sum += d_ar1->Tmat[k][i][j] * Z[j];
            }
            X_next[i] = sum;
        }

        for (int d = 0; d < N_DIM; d++) {
            d_paths[(d * N_STEPS + (k + 1)) * M_PATHS + tid] = X_next[d];
            X_curr[d] = X_next[d];
        }
    }

    d_rng_states[tid] = local_state;
}

static float payoff(float S, float K) {
    return fmaxf(K - S, 0.0f);  // put
}

static void evaluate_basis(float x, int N_BASIS, float* basis_out) {
    // Laguerre polynomials
    basis_out[0] = 1.0f;
    if (N_BASIS > 1) basis_out[1] = 1.0f - x;
    for (int i = 2; i < N_BASIS; i++) {
        basis_out[i] = ((2 * i - 1 - x) * basis_out[i - 1] - (i - 1) * basis_out[i - 2]) / i;
    }
}

float price_LSM(float *h_paths, const SimulationParams *sim) {
    int M = sim->M_PATHS;
    int N = sim->N_STEPS;
    int N_BASIS = sim->N_BASIS;
    float K = sim->K;
    float dT = sim->dT;
    float r = sim->r;

    float* cashflows = (float*)calloc(M * N, sizeof(float));
    bool* exercised = (bool*)calloc(M * N, sizeof(bool));
    int* exercise_count = (int*)calloc(M, sizeof(int));

    // initialize with final payoffs at maturity
    int t_final = N - 1;
    for (int m = 0; m < M; m++) {
        float S = h_paths[(0 * N + t_final) * M + m];
        float val = payoff(S, K);
        cashflows[t_final * M + m] = val;
    }

    // backward induction
    for (int t = N - 2; t >= 0; t--) {
        // paths still eligible to exercise
        int n_train = 0;
        float *X = (float*)malloc(sizeof(float) * M);
        float *Y = (float*)malloc(sizeof(float) * M);

        for (int m = 0; m < M; m++) {
            if (exercise_count[m] >= sim->Q_max) continue;

            float S = h_paths[(0 * N + t) * M + m];
            float immediate = payoff(S, K);
            if (immediate > 0.0f) {
                X[n_train] = S;

                // discounted cashflow
                float discounted = 0.0f;
                for (int t2 = t + 1; t2 < N; t2++) {
                    discounted += DISCOUNT((t2 - t) * dT, r) * cashflows[t2 * M + m];
                }

                Y[n_train] = discounted;
                n_train++;
            }
        }

        // regression
        float **Phi = (float**)malloc(n_train * sizeof(float*));
        float *b = (float*)calloc(N_BASIS, sizeof(float));
        float **A = (float**)calloc(N_BASIS, sizeof(float*));
        for (int i = 0; i < N_BASIS; i++) {
            A[i] = (float*)calloc(N_BASIS, sizeof(float));
        }

        for (int i = 0; i < n_train; i++) {
            Phi[i] = (float*)malloc(N_BASIS * sizeof(float));
            evaluate_basis(X[i], N_BASIS, Phi[i]);
        }

        // A = phi.T * phi and b = phi.T * Y
        for (int i = 0; i < N_BASIS; i++) {
            for (int j = 0; j < N_BASIS; j++) {
                for (int k = 0; k < n_train; k++) {
                    A[i][j] += Phi[k][i] * Phi[k][j];
                }
            }
            for (int k = 0; k < n_train; k++) {
                b[i] += Phi[k][i] * Y[k];
            }
        }

        // A * beta = b using Gaussian elimination
        float *beta = (float*)calloc(N_BASIS, sizeof(float));
        for (int i = 0; i < N_BASIS; i++) beta[i] = b[i];

        // exercise decisions
        for (int m = 0; m < M; m++) {
            if (exercise_count[m] >= sim->Q_max) continue;
            float S = h_paths[(0 * N + t) * M + m];
            float phi[MAX_BASIS];
            evaluate_basis(S, N_BASIS, phi);

            float cont_val = 0.0f;
            for (int i = 0; i < N_BASIS; i++) {
                cont_val += beta[i] * phi[i];
            }

            float immediate = payoff(S, K);
            if (immediate > cont_val) {
                cashflows[t * M + m] = immediate;
                exercised[t * M + m] = true;
                exercise_count[m]++;
            }
        }

        // free mgemory
        for (int i = 0; i < n_train; i++) free(Phi[i]);
        free(Phi); free(X); free(Y); free(beta);
        for (int i = 0; i < N_BASIS; i++) free(A[i]);
        free(A); free(b);
    }

    // Compute present value
    float value = 0.0f;
    for (int m = 0; m < M; m++) {
        for (int t = 0; t < N; t++) {
            float c = cashflows[t * M + m];
            value += DISCOUNT(t * dT, r) * c;
        }
    }
    value /= M;

    free(cashflows);
    free(exercised);
    free(exercise_count);

    return value;
}

void generate_quantization_grids(const SimulationParams *sim, const AR1ModelParams *h_ar1, float *h_Gamma) {
    for (int k = 0; k < sim->N_STEPS; k++) {
        float x_std[2];

        for (int d = 0; d < 2; d++) {
            // std deviation assuming mean 0
            float var = 0.0f;
            for (int j = 0; j < 2; j++) {
                var += h_ar1->Tmat[k][d][j] * h_ar1->Tmat[k][d][j];
            }
            x_std[d] = sqrtf(var);
        }

        // square grid uniformly distributed within 3 std
        int grid_size = sim->N_GRID;
        int side = (int)sqrt(grid_size);
        if (side * side != grid_size) {
            fprintf(stderr, "N_GRID must be a perfect square (e.g., 100, 225, 400)\n");
            exit(1);
        }

        for (int i = 0; i < side; i++) {
            float x1 = -3.0f * x_std[0] + (6.0f * x_std[0]) * i / (side - 1);
            for (int j = 0; j < side; j++) {
                float x2 = -3.0f * x_std[1] + (6.0f * x_std[1]) * j / (side - 1);
                int idx = i * side + j;
                h_Gamma[(k * grid_size + idx) * 2 + 0] = x1;
                h_Gamma[(k * grid_size + idx) * 2 + 1] = x2;
            }
        }
    }
}
  
void run_options_pipeline(const SimulationParams *sim, const AR1ModelParams *h_ar1) {
// memory allocation
    const int m_paths = sim->M_PATHS;
    const int n_steps = sim->N_STEPS;
    const int n_dim = sim->N_DIM;
    const int n_grid = sim->N_GRID;

    size_t path_bytes = sizeof(float) * m_paths * n_steps * n_dim;
    size_t V_bytes = sizeof(float) * m_paths;
    size_t gamma_bytes = sizeof(float) * n_steps * n_grid * n_dim;
    size_t pkij_bytes = sizeof(int) * n_steps * n_grid * n_grid;
    size_t pki_bytes = sizeof(int) * n_steps * n_grid;
    size_t Pkij_bytes = sizeof(float) * n_steps * n_grid * n_grid;

    // host
    float *h_Gamma = NULL; 
    float *h_paths = NULL;

    checkCudaErrors(cudaMallocHost(&h_Gamma, gamma_bytes));
    checkCudaErrors(cudaMallocHost(&h_paths, path_bytes));

    // device
    AR1ModelParams *d_ar1 = NULL;
    float *d_paths = NULL;
    float *d_V = NULL;
    float *d_Gamma = NULL;
    float *d_Pkij = NULL;
    int *d_pkij = NULL;
    int *d_pki = NULL;
    curandState *d_rng_states = NULL;

    checkCudaErrors(cudaMalloc(&d_ar1, sizeof(AR1ModelParams)));                // ar1 sim for device
    cudaMemcpy(d_ar1, h_ar1, sizeof(AR1ModelParams), cudaMemcpyHostToDevice);

    checkCudaErrors(cudaMalloc(&d_paths, path_bytes));      // simulation paths
    checkCudaErrors(cudaMalloc(&d_V, V_bytes));             // value vector
    checkCudaErrors(cudaMalloc(&d_Gamma, gamma_bytes));     // quantization grids
    checkCudaErrors(cudaMalloc(&d_Pkij, Pkij_bytes));       // transition probabilities

    // transition counters for both quatizations
    checkCudaErrors(cudaMalloc(&d_pkij, pkij_bytes));
    checkCudaErrors(cudaMalloc(&d_pki, pki_bytes));
    checkCudaErrors(cudaMemset(d_pkij, 0, pkij_bytes));
    checkCudaErrors(cudaMemset(d_pki, 0, pki_bytes));

    checkCudaErrors(cudaMalloc(&d_rng_states, sizeof(curandState) * m_paths));      // RNG states

// init quatization grid
    generate_quantization_grids(sim, h_ar1, h_Gamma);
    cudaMemcpy(d_Gamma, h_Gamma, gamma_bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((m_paths + block.x - 1) / block.x);

// init rng
    init_rng_kernel<<<grid, block>>>(d_rng_states, sim->RNG_SEED, m_paths);
    checkCudaErrors(cudaDeviceSynchronize());

// Monte Carlo simulation
    simulate_paths_kernel<<<grid, block>>>(
        d_paths, d_ar1, d_rng_states, m_paths, n_steps
    );
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(h_paths, d_paths, path_bytes, cudaMemcpyDeviceToHost);

// Longstaff-Schwartz CPU only
    float V_LSM = price_LSM(h_paths, sim);

// to do
    // Tree Quantization Algorithm II
    // estimate_transition_probabilities_algo2();
    // V_TQ2 = backward_induction_quantized();

    // // Run Tree Quantization Algorithm III
    // estimate_transition_probabilities_algo3();
    // V_TQ3 = backward_induction_quantized();

    // print_results(V_LSM, V_TQ2, V_TQ3);
    // compare_performance();

// clean up
    if (h_Gamma) cudaFreeHost(h_Gamma);

    h_Gamma = NULL;

    if (d_paths) cudaFree(d_paths);
    if (d_V) cudaFree(d_V);
    if (d_Gamma) cudaFree(d_Gamma);
    if (d_pkij) cudaFree(d_pkij);
    if (d_pki) cudaFree(d_pki);
    if (d_Pkij) cudaFree(d_Pkij);
    if (d_rng_states) cudaFree(d_rng_states);

    d_paths = NULL;
    d_V = NULL;
    d_Gamma = NULL;
    d_pkij = NULL;
    d_pki = NULL;
    d_Pkij = NULL;
    d_rng_states = NULL;    
}


void printUsage(char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("Options:                                Defaults:\n");
    printf("  -m  Number of Monte Carlo samples  -  100000\n");
    printf("  -n  Number of time steps - - - - - -  365\n");
    printf("  -g  Grid size per time step  - - - -  400\n");
    printf("  -t  Time (years) to maturity - - - -  1.0\n");
    printf("  -s  Initial stock price  - - - - - -  100.0\n");
    printf("  -k  Strike price - - - - - - - - - -  100.0\n");
    printf("  -r  Risk-free rate - - - - - - - - -  0.05\n");
    printf("  -q  Minimum quantity - - - - - - - -  50\n");
    printf("  -Q  Maximum quantity - - - - - - - -  150\n");
    printf("  -S  Random number generator seed - -  42\n");
    printf("  -w First Gaussian weight alpha1 - -  1.0\n");
    printf("  -x Second Gaussian weight alpha2  -  0.4\n");
    printf("  -y First Gaussian sigma1  - - - - -  0.3\n");
    printf("  -z Second Gaussian sigma2 - - - - -  0.2\n");
}

void printSimulationParams(SimulationParams *sim) {
    printf("Model parameters:\n");
    printf("  Number of Monte Carlo samples  -  %d\n", sim->M_PATHS);
    printf("  Number of time steps - - - - - -  %d\n", sim->N_STEPS);
    printf("  Grid size per time step  - - - -  %d\n", sim->N_GRID);
    printf("  Time (years) to maturity - - - -  %.4f\n", sim->T);
    printf("  Initial stock price  - - - - - -  %.4f\n", sim->S0);
    printf("  Strike price - - - - - - - - - -  %.4f\n", sim->K);
    printf("  Risk-free rate - - - - - - - - -  %.4f\n", sim->r);
    printf("  Minimum quantity - - - - - - - -  %d\n", sim->Q_min);
    printf("  Maximum quantity - - - - - - - -  %d\n", sim->Q_max);
    printf("  Random number generator seed - -  %lu\n", sim->RNG_SEED);
    printf("  First Gaussian weight alpha1 - -  %.4f\n", sim->alpha1);
    printf("  Second Gaussian weight alpha2  -  %.4f\n", sim->alpha2);
    printf("  First Gaussian sigma1  - - - - -  %.4f\n", sim->sigma1);
    printf("  Second Gaussian sigma2 - - - - -  %.4f\n", sim->sigma2);
}

void init_SimulationParams(SimulationParams *sim) {
    sim->M_PATHS = 100000;
    sim->N_STEPS = 365;
    sim->T = 1.0f;
    sim->dT = 1.0f / 365;
    sim->S0 = 100.0f;
    sim->K = 100.0f;
    sim->r = 0.05f;
    sim->Q_min = 50;
    sim->Q_max = 150;
    sim->N_BASIS = 3;
    sim->N_GRID = 400;
    sim->N_DIM = 2;
    sim->RNG_SEED = 42UL;
    sim->alpha1 = 1.0f;
    sim->alpha2 = 0.4f;
    sim->sigma1 = 0.3f;
    sim->sigma2 = 0.2f;
}

void init_AR1ModelParams(AR1ModelParams *ar1, SimulationParams *sim) {
    for (int d = 0; d < MAX_DIM; d++) {
        ar1->X0[d] = 0.0f;
    }

    const float dT = sim->dT;
    const float alpha1 = sim->alpha1, alpha2 = sim->alpha2;
    const float sigma1 = sim->sigma1, sigma2 = sim->sigma2;
    const int n_steps = sim->N_STEPS;

    for (int k = 0; k < n_steps; k++) {
        float e1 = expf(-alpha1 * dT);
        float e2 = expf(-alpha2 * dT);

        ar1->A[k][0][0] = e1;
        ar1->A[k][0][1] = 0.0f;
        ar1->A[k][1][0] = 0.0f;
        ar1->A[k][1][1] = e2;

        float var1 = sigma1 * sqrtf((1 - expf(-2.0f * alpha1 * dT)) / (2.0f * alpha1));
        float var2 = sigma2 * sqrtf((1 - expf(-2.0f * alpha2 * dT)) / (2.0f * alpha2));

        ar1->Tmat[k][0][0] = var1;
        ar1->Tmat[k][0][1] = 0.0f;
        ar1->Tmat[k][1][0] = 0.0f;
        ar1->Tmat[k][1][1] = var2;
    }
}


int main(int argc, char **argv) {
    SimulationParams *sim = (SimulationParams*) malloc(sizeof(SimulationParams));
    init_SimulationParams(sim);

    int opt;
    int val_i;
    float val_f;
    unsigned long val_ul;
    while ((opt = getopt(argc, argv, "m:n:g:t:s:k:r:q:Q:S:w:x:y:z:")) != -1) {
        switch (opt) {
            case 'm':
                val_i = atoi(optarg);
                if (val_i > 0) sim->M_PATHS = val_i; 
                break;
            case 'n':             
                val_i = atoi(optarg);
                if (val_i > 0 && val_i < MAX_STEPS) sim->N_STEPS = val_i; 
                break;
            case 'g': 
                val_i = atoi(optarg);
                if (val_i > 0 && val_i < MAX_GRID) sim->N_GRID = val_i; 
                break;
            case 't':
                val_f = atof(optarg);
                if (val_f > 0.0f) sim->T = val_f; 
                break;
            case 's':
                val_f = atof(optarg);
                if (val_f > 0.0f) sim->S0 = val_f; 
                break;
            case 'k':
                val_f = atof(optarg);
                if (val_f > 0.0f) sim->K = val_f; 
                break;
            case 'r':
                val_f = atof(optarg);
                if (val_f > 0.0f) sim->r = val_f; 
                break;
            case 'q':
                val_i = atoi(optarg);
                if (val_i > 0) sim->Q_min = val_i; 
                break;
            case 'Q':
                val_i = atoi(optarg);
                if (val_i > 0) sim->Q_max = val_i; 
                break;
            case 'S':
                val_ul = strtoul(optarg, NULL, 10);
                if (val_ul > 0UL) sim->RNG_SEED = val_ul;
                break;
            case 'w':
                val_f = atof(optarg);
                if (val_f > 0.0f) sim->alpha1 = val_f; 
                break;
            case 'x':
                val_f = atof(optarg);
                if (val_f > 0.0f) sim->alpha2 = val_f; 
                break;
            case 'y':
                val_f = atof(optarg);
                if (val_f > 0.0f) sim->sigma1 = val_f; 
                break;
            case 'z':
                val_f = atof(optarg);
                if (val_f > 0.0f) sim->sigma2 = val_f; 
                break;
            default:
                printUsage(argv[0]);
                free(sim);
                exit(EXIT_FAILURE);
        }
    }

    AR1ModelParams *ar1 = (AR1ModelParams*) malloc(sizeof(AR1ModelParams));
    init_AR1ModelParams(ar1, sim);

    printSimulationParams(sim);

    run_options_pipeline(sim, ar1);

    free(sim);
    free(ar1);
    return EXIT_SUCCESS;
}
