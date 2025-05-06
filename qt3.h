#ifndef QT3_H
#define QT3_H

unsigned long long total_flops_qt3();
void generate_grid_qt3(float *grid, int size, float mean, float stddev);
void backward_induction_qt3(float *grid, int *p_ij, int *p_i, float *values);
extern "C" void qt3();

__global__ void compute_transitions_qt3(int *d_p_ij, int *d_p_i, float *d_grid, 
                                        int num_timesteps, int num_paths, int grid_size,
                                        float stddev);

#endif
