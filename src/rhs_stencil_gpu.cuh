#ifndef RHS_STENCIL_GPU_CUH
#define RHS_STENCIL_GPU_CUH

// RHS-only stencil helpers live in the caller's translation unit so nvcc can
// inline them even when the executable uses relocatable device code.

__device__ __forceinline__ double rhs_symmetry_load(
    int ord, int ex0, int ex1, int ex2, const double* func,
    int i1b, int j1b, int k1b,
    double sym1, double sym2, double sym3
) {
    if (i1b < -ord + 1 || i1b > ex0) return 0.0;
    if (j1b < -ord + 1 || j1b > ex1) return 0.0;
    if (k1b < -ord + 1 || k1b > ex2) return 0.0;

    int ii = i1b;
    int jj = j1b;
    int kk = k1b;
    double factor = 1.0;
    if (ii <= 0) { ii = 1 - ii; factor *= sym1; }
    if (jj <= 0) { jj = 1 - jj; factor *= sym2; }
    if (kk <= 0) { kk = 1 - kk; factor *= sym3; }

    if (ii < 1 || ii > ex0) return 0.0;
    if (jj < 1 || jj > ex1) return 0.0;
    if (kk < 1 || kk > ex2) return 0.0;

    return func[((kk - 1) * ex1 + (jj - 1)) * ex0 + (ii - 1)] * factor;
}

#endif
