#ifndef LOPSIDEDIFF_H
#define LOPSIDEDIFF_H

#ifdef USE_GPU
#include <cuda_runtime.h>

__device__ double d_lopsided_point(
    const int ex[3], const double* f,
    double vx, double vy, double vz,
    double d12dx, double d12dy, double d12dz,
    int imin, int jmin, int kmin, int imax, int jmax, int kmax,
    int symmetry, double SYM1, double SYM2, double SYM3,
    int i, int j, int k
);
#endif

#endif /* LOPSIDEDIFF_H */
