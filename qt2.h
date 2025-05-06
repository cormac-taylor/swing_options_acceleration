#ifndef QT2_H
#define QT2_H

void generate_grid_qt2(float *grid, int size, float mean, float stddev);
float grid_to_price_qt2(float x, float t);
void backward_induction_qt2(float *grid, int *p_ij, int *p_i, float *values);
extern "C" void qt2();

__global__ void compute_transitions_qt2(int *d_p_ij, int *d_p_i, float *d_grid, 
                                        int num_timesteps, int num_paths, int grid_size);

#endif
