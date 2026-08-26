#include "kodiss.h"

#include "fmisc.h"
#include "rhs_stencil_gpu.cuh"

#include "macrodef.fh"
#include <cmath>

__device__ double d_kodis_point(
    const int ex[3], const double* f,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, double eps,
    int i, int j, int k // 0-based
) {
    const double ONE = 1.0;
    const double SIX = 6.0;
    const double FIT = 15.0;
    const double TWT = 20.0;
    const double cof = 64.0;
    const int NO_SYMM = 0, OCTANT = 2;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const int ex0 = ex[0], ex1 = ex[1], ex2 = ex[2];
    const int imax = ex0 - 1;
    const int jmax = ex1 - 1;
    const int kmax = ex2 - 1;

    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -3;
    if (symmetry == OCTANT && fabs(X[0]) < dX) imin = -3;
    if (symmetry == OCTANT && fabs(Y[0]) < dY) jmin = -3;


#define fh(ii, jj, kk) rhs_symmetry_load(3, ex0, ex1, ex2, f, (ii) + 1, (jj) + 1, (kk) + 1, SYM1, SYM2, SYM3)

    double rhs_add = 0.0;

    if (i - 3 >= imin && i + 3 <= imax &&
        j - 3 >= jmin && j + 3 <= jmax &&
        k - 3 >= kmin && k + 3 <= kmax) {

        rhs_add = eps / cof * (
            ((fh(i-3,j,k) + fh(i+3,j,k)) - SIX*(fh(i-2,j,k) + fh(i+2,j,k)) +
             FIT*(fh(i-1,j,k) + fh(i+1,j,k)) - TWT*fh(i,j,k)) / dX +
            ((fh(i,j-3,k) + fh(i,j+3,k)) - SIX*(fh(i,j-2,k) + fh(i,j+2,k)) +
             FIT*(fh(i,j-1,k) + fh(i,j+1,k)) - TWT*fh(i,j,k)) / dY +
            ((fh(i,j,k-3) + fh(i,j,k+3)) - SIX*(fh(i,j,k-2) + fh(i,j,k+2)) +
             FIT*(fh(i,j,k-1) + fh(i,j,k+1)) - TWT*fh(i,j,k)) / dZ
        );
    }

#undef fh
    return rhs_add;
}