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

typedef struct {
    // Simulation control
    int M_PATHS;       // Number of Monte Carlo paths
    int N_STEPS;       // Number of time steps
    float T;           // Maturity (in years)
    float dT;          // Time step size = T / N_STEPS

    // Option parameters
    float S0;          // Initial asset price
    float K;           // Strike price
    float r;           // Risk-free rate
    int Q_min, Q_max;  // Min and max exercise rights

    // Longstaff-Schwartz
    int N_BASIS;       // Number of basis functions for regression

    // Quantization
    int N_GRID;        // Number of grid points per time step
    int N_DIM;         // State space dimensionality (1 for GBM, 2 for OU)

    // Random number generation
    unsigned long RNG_SEED;

    // AR(1) 
    float alpha1;
    float alpha2;
    float sigma1;
    float sigma2;
} SimulationParams;

typedef struct {
    float X0[MAX_DIM];                          // Initial state
    float A[MAX_STEPS][MAX_DIM][MAX_DIM];       // A_k transition matrices
    float Tmat[MAX_STEPS][MAX_DIM][MAX_DIM];    // T_k volatility matrices
} AR1ModelParams;


void run_options_pipeline(SimulationParams *sim, AR1ModelParams *ar1) {
    // init_parameters();
    // allocate_memory();

    // generate_quantization_grids();  // Shared between TQ-II and TQ-III

    // // Monte Carlo simulation
    // simulate_paths();  // For LSM (used directly), for TQ (used for transition estimation)

    // // Run Longstaff-Schwartz
    // V_LSM = price_LSM();

    // // Run Tree Quantization Algorithm II
    // estimate_transition_probabilities_algo2();
    // V_TQ2 = backward_induction_quantized();

    // // Run Tree Quantization Algorithm III
    // estimate_transition_probabilities_algo3();
    // V_TQ3 = backward_induction_quantized();

    // print_results(V_LSM, V_TQ2, V_TQ3);
    // compare_performance();
    // cleanup();
}


void printUsage(char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("Options:                                defaults:\n");
    printf("  -m  Number of Monte Carlo samples  -  100000\n");
    printf("  -n  Number of time steps - - - - - -  365\n");
    printf("  -g  Grid size per time step  - - - -  250\n");
    printf("  -t  Time (years) to maturity - - - -  1.0\n");
    printf("  -s  Initial stock price  - - - - - -  100.0\n");
    printf("  -k  Strike price - - - - - - - - - -  100.0\n");
    printf("  -r  Risk-free rate - - - - - - - - -  0.05\n");
    printf("  -q  Minimum quantity - - - - - - - -  50\n");
    printf("  -Q  Maximum quantity - - - - - - - -  150\n");
    printf("  -S  Random number generator seed - -  42\n");

    printf("Put the following options last:\n");
    printf("  -a1 First Gaussian weight alpha1 - -  1.0\n");
    printf("  -a2 Second Gaussian weight alpha2  -  0.4\n");
    printf("  -s1 First Gaussian sigma1  - - - - -  0.3\n");
    printf("  -s2 Second Gaussian sigma2 - - - - -  0.2\n");
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
    sim->N_GRID = 250;
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

        // A_k = diag(e1, e2)
        ar1->A[k][0][0] = e1;
        ar1->A[k][0][1] = 0.0f;
        ar1->A[k][1][0] = 0.0f;
        ar1->A[k][1][1] = e2;

        // T_k = sqrt(Var) = diag(sigma_i * sqrt((1 - e^{-2*alpha_i*delta_t}) / (2*alpha_i)))
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
    while ((opt = getopt(argc, argv, "m:n:g:t:s:k:r:q:Q:S:")) != -1) {
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
            default:
                printUsage(argv[0]);
                free(sim);
                exit(EXIT_FAILURE);
        }
    }

    for (int i = optind; i < argc; i++) {
        bool param_idx_valid = i + 1 < argc;
        if (!strcmp(argv[i], "-a1") && param_idx_valid) {
            sim->alpha1 = atof(argv[++i]);
        } else if (!strcmp(argv[i], "-a2") && param_idx_valid) {
            sim->alpha2 = atof(argv[++i]);
        } else if (!strcmp(argv[i], "-s1") && param_idx_valid) {
            sim->sigma1 = atof(argv[++i]);
        } else if (!strcmp(argv[i], "-s2") && param_idx_valid) {
            sim->sigma2 = atof(argv[++i]);
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            printUsage(argv[0]);
            free(sim);
            exit(EXIT_FAILURE);
        }
    }


    // #define MAX_STEPS   366
    // #define MAX_GRID    500


    AR1ModelParams *ar1 = (AR1ModelParams*) malloc(sizeof(AR1ModelParams));
    init_AR1ModelParams(ar1, sim);

    printSimulationParams(sim);

    run_options_pipeline(sim, ar1);

    free(sim);
    free(ar1);
    return EXIT_SUCCESS;
}
