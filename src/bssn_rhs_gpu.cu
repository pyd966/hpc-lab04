#include "bssn_rhs.h"

#include "fmisc.h"
#include "derivatives.h"
#include "kodiss.h"
#include "lopsidediff.h"
#include "gpu_manager.h"
#include "advection_compact_gpu.cuh"
#include "hessian_compact_gpu.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <iostream>

// ==========================================
// 宏定义 (对应 macrodef.fh 和 bssn_rhs.f90)
// ==========================================
// Fortran Column-Major Layout: x varies fastest
#define IDX3D(i, j, k, nx, ny, nz) ((i) + (nx) * ((j) + (ny) * (k)))

constexpr double SYM = 1.0;
constexpr double ANTI = -1.0;
constexpr double ZEO = 0.0;
constexpr double ONE = 1.0;
constexpr double TWO = 2.0;
constexpr double FOUR = 4.0;
constexpr double EIGHT = 8.0;
constexpr double PI = M_PI;
constexpr double F1o3 = 1.0 / 3.0;
constexpr double F2o3 = 2.0 / 3.0;
constexpr double F3o2 = 1.5;
constexpr double HALF = 0.5;
constexpr double FF = 0.75;
constexpr double eta = 2.0;
constexpr double F8 = 8.0;
constexpr double F16 = 16.0;

__global__ void rhs_evolution_kernel(
    int ex0, int ex1, int ex2, double T, double* X, double* Y, double* Z,
    double* chi, double* trK,
    double* dxx, double* gxy, double* gxz,
    double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz,
    double* Ayy, double* Ayz, double* Azz,
    double* Gamx, double* Gamy, double* Gamz,
    double* Lap,
    double* betax, double* betay, double* betaz,
    double* dtSfx, double* dtSfy, double* dtSfz,
    double* chi_rhs, double* trK_rhs,
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs,
    double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs,
    double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs,
    double* Lap_rhs,
    double* betax_rhs, double* betay_rhs, double* betaz_rhs,
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
    double* rho, double* Sx, double* Sy, double* Sz,
    double* Sxx, double* Sxy, double* Sxz,
    double* Syy, double* Syz, double* Szz,
    double* Gamxxx, double* Gamxxy, double* Gamxxz,
    double* Gamxyy, double* Gamxyz, double* Gamxzz,
    double* Gamyxx, double* Gamyxy, double* Gamyxz,
    double* Gamyyy, double* Gamyyz, double* Gamyzz,
    double* Gamzxx, double* Gamzxy, double* Gamzxz,
    double* Gamzyy, double* Gamzyz, double* Gamzzz,
    double* Rxx, double* Rxy, double* Rxz,
    double* Ryy, double* Ryz, double* Rzz,
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};

    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];
    double fxx, fxy, fxz, fyy, fyz, fzz;

    d_fdderivs_point(dims, dxx, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz,
                     X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    Rxx[idx] = gupxx * fxx + gupyy * fyy + gupzz * fzz +
               TWO * (gupxy * fxy + gupxz * fxz + gupyz * fyz);
    d_fdderivs_point(dims, dyy, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz,
                     X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    Ryy[idx] = gupxx * fxx + gupyy * fyy + gupzz * fzz +
               TWO * (gupxy * fxy + gupxz * fxz + gupyz * fyz);
    d_fdderivs_point(dims, dzz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz,
                     X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    Rzz[idx] = gupxx * fxx + gupyy * fyy + gupzz * fzz +
               TWO * (gupxy * fxy + gupxz * fxz + gupyz * fyz);
    d_fdderivs_point(dims, gxy, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz,
                     X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    Rxy[idx] = gupxx * fxx + gupyy * fyy + gupzz * fzz +
               TWO * (gupxy * fxy + gupxz * fxz + gupyz * fyz);
    d_fdderivs_point(dims, gxz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz,
                     X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    Rxz[idx] = gupxx * fxx + gupyy * fyy + gupzz * fzz +
               TWO * (gupxy * fxy + gupxz * fxz + gupyz * fyz);
    d_fdderivs_point(dims, gyz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz,
                     X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    Ryz[idx] = gupxx * fxx + gupyy * fyy + gupzz * fzz +
               TWO * (gupxy * fxy + gupxz * fxz + gupyz * fyz);
}


#define RHS_KERNEL_PARAMS \
    int ex0, int ex1, int ex2, double T, double* X, double* Y, double* Z, \
    double* chi, double* trK, \
    double* dxx, double* gxy, double* gxz, \
    double* dyy, double* gyz, double* dzz, \
    double* Axx, double* Axy, double* Axz, \
    double* Ayy, double* Ayz, double* Azz, \
    double* Gamx, double* Gamy, double* Gamz, double* Lap, \
    double* betax, double* betay, double* betaz, \
    double* dtSfx, double* dtSfy, double* dtSfz, \
    double* chi_rhs, double* trK_rhs, \
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs, \
    double* gyy_rhs, double* gyz_rhs, double* gzz_rhs, \
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs, \
    double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs, \
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs, double* Lap_rhs, \
    double* betax_rhs, double* betay_rhs, double* betaz_rhs, \
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs, \
    double* rho, double* Sx, double* Sy, double* Sz, \
    double* Sxx, double* Sxy, double* Sxz, double* Syy, double* Syz, double* Szz, \
    double* Gamxxx, double* Gamxxy, double* Gamxxz, \
    double* Gamxyy, double* Gamxyz, double* Gamxzz, \
    double* Gamyxx, double* Gamyxy, double* Gamyxz, \
    double* Gamyyy, double* Gamyyz, double* Gamyzz, \
    double* Gamzxx, double* Gamzxy, double* Gamzxz, \
    double* Gamzyy, double* Gamzyz, double* Gamzzz, \
    double* Rxx, double* Rxy, double* Rxz, double* Ryy, double* Ryz, double* Rzz, \
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res, \
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res, \
    int symmetry, int lev, double eps, int co

__global__ void rhs_beta_gamma_prepare_kernel(RHS_KERNEL_PARAMS) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];

    double hxx, hxy, hxz, hyy, hyz, hzz;
    d_fdderivs_point(dims, betax, &hxx, &hxy, &hxz, &hyy, &hyz, &hzz,
                     X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    double div_hess_x = hxx;
    double div_hess_y = hxy;
    double div_hess_z = hxz;
    const double lap_betax = gupxx*hxx + gupyy*hyy + gupzz*hzz
                           + TWO*(gupxy*hxy + gupxz*hxz + gupyz*hyz);

    d_fdderivs_point(dims, betay, &hxx, &hxy, &hxz, &hyy, &hyz, &hzz,
                     X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    div_hess_x += hxy;
    div_hess_y += hyy;
    div_hess_z += hyz;
    const double lap_betay = gupxx*hxx + gupyy*hyy + gupzz*hzz
                           + TWO*(gupxy*hxy + gupxz*hxz + gupyz*hyz);

    d_fdderivs_point(dims, betaz, &hxx, &hxy, &hxz, &hyy, &hyz, &hzz,
                     X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);
    div_hess_x += hxz;
    div_hess_y += hyz;
    div_hess_z += hzz;
    const double lap_betaz = gupxx*hxx + gupyy*hyy + gupzz*hzz
                           + TWO*(gupxy*hxy + gupxz*hxz + gupyz*hyz);

    ham_Res[idx] = div_hess_x;
    movx_Res[idx] = div_hess_y;
    movy_Res[idx] = div_hess_z;
    movz_Res[idx] = lap_betax;
    Gmx_Res[idx] = lap_betay;
    Gmy_Res[idx] = lap_betaz;

    const double Gamxa = gupxx*Gamxxx[idx] + gupyy*Gamxyy[idx] + gupzz*Gamxzz[idx]
                       + TWO*(gupxy*Gamxxy[idx] + gupxz*Gamxxz[idx] + gupyz*Gamxyz[idx]);
    const double Gamya = gupxx*Gamyxx[idx] + gupyy*Gamyyy[idx] + gupzz*Gamyzz[idx]
                       + TWO*(gupxy*Gamyxy[idx] + gupxz*Gamyxz[idx] + gupyz*Gamyyz[idx]);
    const double Gamza = gupxx*Gamzxx[idx] + gupyy*Gamzyy[idx] + gupzz*Gamzzz[idx]
                       + TWO*(gupxy*Gamzxy[idx] + gupxz*Gamzxz[idx] + gupyz*Gamzyz[idx]);
    Ayy_rhs[idx] = Gamxa;
    Ayz_rhs[idx] = Gamya;
    Azz_rhs[idx] = Gamza;
}

__global__ void rhs_beta_gamma_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)chi_rhs; (void)trK_rhs; (void)Lap_rhs;
    (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const double betazx = Axx_rhs[idx], betazy = Axy_rhs[idx], betazz = Axz_rhs[idx];
    const double betaxx = gxx_rhs[idx], betaxy = gxy_rhs[idx], betaxz = gxz_rhs[idx];
    const double betayx = gyy_rhs[idx], betayy = gyz_rhs[idx], betayz = gzz_rhs[idx];
    const double div_beta = betaxx + betayy + betazz;
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];

    double val_Gamx_rhs = Gamx_rhs[idx], val_Gamy_rhs = Gamy_rhs[idx], val_Gamz_rhs = Gamz_rhs[idx];
    const double fxx = ham_Res[idx], fxy = movx_Res[idx], fxz = movy_Res[idx];
    const double Gamxa = Ayy_rhs[idx], Gamya = Ayz_rhs[idx], Gamza = Azz_rhs[idx];

    val_Gamx_rhs += F2o3 * Gamxa * div_beta - (Gamxa * betaxx + Gamya * betaxy + Gamza * betaxz)
                  + F1o3 * (gupxx * fxx + gupxy * fxy + gupxz * fxz)
                  + movz_Res[idx];
    val_Gamy_rhs += F2o3 * Gamya * div_beta - (Gamxa * betayx + Gamya * betayy + Gamza * betayz)
                  + F1o3 * (gupxy * fxx + gupyy * fxy + gupyz * fxz)
                  + Gmx_Res[idx];
    val_Gamz_rhs += F2o3 * Gamza * div_beta - (Gamxa * betazx + Gamya * betazy + Gamza * betazz)
                  + F1o3 * (gupxz * fxx + gupyz * fxy + gupzz * fxz)
                  + Gmy_Res[idx];

    Gamx_rhs[idx] = val_Gamx_rhs;
    Gamy_rhs[idx] = val_Gamy_rhs;
    Gamz_rhs[idx] = val_Gamz_rhs;

    // Reuse consumed beta/Gamma scratch to publish derivatives for both Ricci consumers.
    const int dims[3] = {ex0, ex1, ex2};
    double dGamxx, dGamxy, dGamxz;
    double dGamyx, dGamyy, dGamyz;
    double dGamzx, dGamzy, dGamzz;
    d_fderivs_point(dims, Gamx, &dGamxx, &dGamxy, &dGamxz,
                    X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Gamy, &dGamyx, &dGamyy, &dGamyz,
                    X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Gamz, &dGamzx, &dGamzy, &dGamzz,
                    X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

    ham_Res[idx] = dGamxx;
    movx_Res[idx] = dGamxy;
    movy_Res[idx] = dGamxz;
    movz_Res[idx] = dGamyx;
    Gmx_Res[idx] = dGamyy;
    Gmy_Res[idx] = dGamyz;
    Gmz_Res[idx] = dGamzx;
    Ayy_rhs[idx] = dGamzy;
    Ayz_rhs[idx] = dGamzz;
}
__global__ void rhs_ricci_connection_diag_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)chi; (void)trK; (void)Axx; (void)Axy; (void)Axz; (void)Ayy; (void)Ayz; (void)Azz;
    (void)Lap; (void)betax; (void)betay; (void)betaz; (void)dtSfx; (void)dtSfy; (void)dtSfz;
    (void)chi_rhs; (void)trK_rhs; (void)gxx_rhs; (void)gxy_rhs; (void)gxz_rhs; (void)gyy_rhs; (void)gyz_rhs; (void)gzz_rhs;
    (void)Axx_rhs; (void)Axy_rhs; (void)Axz_rhs; (void)Azz_rhs;
    (void)Gamx_rhs; (void)Gamy_rhs; (void)Gamz_rhs; (void)Lap_rhs;
    (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const double l_gxx = dxx[idx] + ONE, l_gxy = gxy[idx], l_gxz = gxz[idx];
    const double l_gyy = dyy[idx] + ONE, l_gyz = gyz[idx], l_gzz = dzz[idx] + ONE;
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];
    const double l_Gamxxx = Gamxxx[idx], l_Gamxxy = Gamxxy[idx], l_Gamxxz = Gamxxz[idx];
    const double l_Gamxyy = Gamxyy[idx], l_Gamxyz = Gamxyz[idx], l_Gamxzz = Gamxzz[idx];
    const double l_Gamyxx = Gamyxx[idx], l_Gamyxy = Gamyxy[idx], l_Gamyxz = Gamyxz[idx];
    const double l_Gamyyy = Gamyyy[idx], l_Gamyyz = Gamyyz[idx], l_Gamyzz = Gamyzz[idx];
    const double l_Gamzxx = Gamzxx[idx], l_Gamzxy = Gamzxy[idx], l_Gamzxz = Gamzxz[idx];
    const double l_Gamzyy = Gamzyy[idx], l_Gamzyz = Gamzyz[idx], l_Gamzzz = Gamzzz[idx];
    double l_Rxx = Rxx[idx], l_Ryy = Ryy[idx], l_Rzz = Rzz[idx];
    double gxxx, gxxy, gxxz, gxyx, gxyy, gxyz, gxzx, gxzy, gxzz;
    double gyyx, gyyy, gyyz, gyzx, gyzy, gyzz, gzzx, gzzy, gzzz;
    // __DIAG_METRIC__
    gxxx = l_gxx * l_Gamxxx + l_gxy * l_Gamyxx + l_gxz * l_Gamzxx;
    gxyx = l_gxx * l_Gamxxy + l_gxy * l_Gamyxy + l_gxz * l_Gamzxy;
    gxzx = l_gxx * l_Gamxxz + l_gxy * l_Gamyxz + l_gxz * l_Gamzxz;
    gyyx = l_gxx * l_Gamxyy + l_gxy * l_Gamyyy + l_gxz * l_Gamzyy;
    gyzx = l_gxx * l_Gamxyz + l_gxy * l_Gamyyz + l_gxz * l_Gamzyz;
    gzzx = l_gxx * l_Gamxzz + l_gxy * l_Gamyzz + l_gxz * l_Gamzzz;

    gxxy = l_gxy * l_Gamxxx + l_gyy * l_Gamyxx + l_gyz * l_Gamzxx;
    gxyy = l_gxy * l_Gamxxy + l_gyy * l_Gamyxy + l_gyz * l_Gamzxy;
    gxzy = l_gxy * l_Gamxxz + l_gyy * l_Gamyxz + l_gyz * l_Gamzxz;
    gyyy = l_gxy * l_Gamxyy + l_gyy * l_Gamyyy + l_gyz * l_Gamzyy;
    gyzy = l_gxy * l_Gamxyz + l_gyy * l_Gamyyz + l_gyz * l_Gamzyz;
    gzzy = l_gxy * l_Gamxzz + l_gyy * l_Gamyzz + l_gyz * l_Gamzzz;

    gxxz = l_gxz * l_Gamxxx + l_gyz * l_Gamyxx + l_gzz * l_Gamzxx;
    gxyz = l_gxz * l_Gamxxy + l_gyz * l_Gamyxy + l_gzz * l_Gamzxy;
    gxzz = l_gxz * l_Gamxxz + l_gyz * l_Gamyxz + l_gzz * l_Gamzxz;
    gyyz = l_gxz * l_Gamxyy + l_gyz * l_Gamyyy + l_gzz * l_Gamzyy;
    gyzz = l_gxz * l_Gamxyz + l_gyz * l_Gamyyz + l_gzz * l_Gamzyz;
    gzzz = l_gxz * l_Gamxzz + l_gyz * l_Gamyzz + l_gzz * l_Gamzzz;
    const double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz +
                         TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    const double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz +
                         TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    const double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz +
                         TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);
    // __DIAG_FORMULAS__
    // Rxx Correction
    l_Rxx = -HALF * l_Rxx +
          l_gxx * ham_Res[idx] + l_gxy * movz_Res[idx] + l_gxz * Gmz_Res[idx] +
          Gamxa * gxxx + Gamya * gxyx + Gamza * gxzx +
          gupxx * (TWO*(l_Gamxxx*gxxx + l_Gamyxx*gxyx + l_Gamzxx*gxzx) + l_Gamxxx*gxxx + l_Gamyxx*gxxy + l_Gamzxx*gxxz) +
          gupxy * (TWO*(l_Gamxxx*gxyx + l_Gamyxx*gyyx + l_Gamzxx*gyzx + l_Gamxxy*gxxx + l_Gamyxy*gxyx + l_Gamzxy*gxzx) + l_Gamxxy*gxxx + l_Gamyxy*gxxy + l_Gamzxy*gxxz + l_Gamxxx*gxyx + l_Gamyxx*gxyy + l_Gamzxx*gxyz) +
          gupxz * (TWO*(l_Gamxxx*gxzx + l_Gamyxx*gyzx + l_Gamzxx*gzzx + l_Gamxxz*gxxx + l_Gamyxz*gxyx + l_Gamzxz*gxzx) + l_Gamxxz*gxxx + l_Gamyxz*gxxy + l_Gamzxz*gxxz + l_Gamxxx*gxzx + l_Gamyxx*gxzy + l_Gamzxx*gxzz) +
          gupyy * (TWO*(l_Gamxxy*gxyx + l_Gamyxy*gyyx + l_Gamzxy*gyzx) + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz) +
          gupyz * (TWO*(l_Gamxxy*gxzx + l_Gamyxy*gyzx + l_Gamzxy*gzzx + l_Gamxxz*gxyx + l_Gamyxz*gyyx + l_Gamzxz*gyzx) + l_Gamxxz*gxyx + l_Gamyxz*gxyy + l_Gamzxz*gxyz + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz) +
          gupzz * (TWO*(l_Gamxxz*gxzx + l_Gamyxz*gyzx + l_Gamzxz*gzzx) + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz);

    // Ryy Correction
    l_Ryy = -HALF * l_Ryy +
          l_gxy * movx_Res[idx] + l_gyy * Gmx_Res[idx] + l_gyz * Ayy_rhs[idx] +
          Gamxa * gxyy + Gamya * gyyy + Gamza * gyzy +
          gupxx * (TWO*(l_Gamxxy*gxxy + l_Gamyxy*gxyy + l_Gamzxy*gxzy) + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz) +
          gupxy * (TWO*(l_Gamxxy*gxyy + l_Gamyxy*gyyy + l_Gamzxy*gyzy + l_Gamxyy*gxxy + l_Gamyyy*gxyy + l_Gamzyy*gxzy) + l_Gamxyy*gxyx + l_Gamyyy*gxyy + l_Gamzyy*gxyz + l_Gamxxy*gyyx + l_Gamyxy*gyyy + l_Gamzxy*gyyz) +
          gupxz * (TWO*(l_Gamxxy*gxzy + l_Gamyxy*gyzy + l_Gamzxy*gzzy + l_Gamxyz*gxxy + l_Gamyyz*gxyy + l_Gamzyz*gxzy) + l_Gamxyz*gxyx + l_Gamyyz*gxyy + l_Gamzyz*gxyz + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) +
          gupyy * (TWO*(l_Gamxyy*gxyy + l_Gamyyy*gyyy + l_Gamzyy*gyzy) + l_Gamxyy*gyyx + l_Gamyyy*gyyy + l_Gamzyy*gyyz) +
          gupyz * (TWO*(l_Gamxyy*gxzy + l_Gamyyy*gyzy + l_Gamzyy*gzzy + l_Gamxyz*gxyy + l_Gamyyz*gyyy + l_Gamzyz*gyzy) + l_Gamxyz*gyyx + l_Gamyyz*gyyy + l_Gamzyz*gyyz + l_Gamxyy*gyzx + l_Gamyyy*gyzy + l_Gamzyy*gyzz) +
          gupzz * (TWO*(l_Gamxyz*gxzy + l_Gamyyz*gyzy + l_Gamzyz*gzzy) + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz);

    // Rzz Correction
    l_Rzz = -HALF * l_Rzz +
          l_gxz * movy_Res[idx] + l_gyz * Gmy_Res[idx] + l_gzz * Ayz_rhs[idx] +
          Gamxa * gxzz + Gamya * gyzz + Gamza * gzzz +
          gupxx * (TWO*(l_Gamxxz*gxxz + l_Gamyxz*gxyz + l_Gamzxz*gxzz) + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz) +
          gupxy * (TWO*(l_Gamxxz*gxyz + l_Gamyxz*gyyz + l_Gamzxz*gyzz + l_Gamxyz*gxxz + l_Gamyyz*gxyz + l_Gamzyz*gxzz) + l_Gamxyz*gxzx + l_Gamyyz*gxzy + l_Gamzyz*gxzz + l_Gamxxz*gyzx + l_Gamyxz*gyzy + l_Gamzxz*gyzz) +
          gupxz * (TWO*(l_Gamxxz*gxzz + l_Gamyxz*gyzz + l_Gamzxz*gzzz + l_Gamxzz*gxxz + l_Gamyzz*gxyz + l_Gamzzz*gxzz) + l_Gamxzz*gxzx + l_Gamyzz*gxzy + l_Gamzzz*gxzz + l_Gamxxz*gzzx + l_Gamyxz*gzzy + l_Gamzxz*gzzz) +
          gupyy * (TWO*(l_Gamxyz*gxyz + l_Gamyyz*gyyz + l_Gamzyz*gyzz) + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz) +
          gupyz * (TWO*(l_Gamxyz*gxzz + l_Gamyyz*gyzz + l_Gamzyz*gzzz + l_Gamxzz*gxyz + l_Gamyzz*gyyz + l_Gamzzz*gyzz) + l_Gamxzz*gyzx + l_Gamyzz*gyzy + l_Gamzzz*gyzz + l_Gamxyz*gzzx + l_Gamyyz*gzzy + l_Gamzyz*gzzz) +
          gupzz * (TWO*(l_Gamxzz*gxzz + l_Gamyzz*gyzz + l_Gamzzz*gzzz) + l_Gamxzz*gzzx + l_Gamyzz*gzzy + l_Gamzzz*gzzz);
    Rxx[idx] = l_Rxx; Ryy[idx] = l_Ryy; Rzz[idx] = l_Rzz;
}
__global__ void rhs_ricci_connection_offdiag_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)chi; (void)trK; (void)Axx; (void)Axy; (void)Axz; (void)Ayy; (void)Ayz; (void)Azz;
    (void)Lap; (void)betax; (void)betay; (void)betaz; (void)dtSfx; (void)dtSfy; (void)dtSfz;
    (void)chi_rhs; (void)trK_rhs; (void)gxx_rhs; (void)gxy_rhs; (void)gxz_rhs; (void)gyy_rhs; (void)gyz_rhs; (void)gzz_rhs;
    (void)Axx_rhs; (void)Axy_rhs; (void)Axz_rhs; (void)Azz_rhs;
    (void)Gamx_rhs; (void)Gamy_rhs; (void)Gamz_rhs; (void)Lap_rhs;
    (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const double l_gxx = dxx[idx] + ONE, l_gxy = gxy[idx], l_gxz = gxz[idx];
    const double l_gyy = dyy[idx] + ONE, l_gyz = gyz[idx], l_gzz = dzz[idx] + ONE;
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];
    const double l_Gamxxx = Gamxxx[idx], l_Gamxxy = Gamxxy[idx], l_Gamxxz = Gamxxz[idx];
    const double l_Gamxyy = Gamxyy[idx], l_Gamxyz = Gamxyz[idx], l_Gamxzz = Gamxzz[idx];
    const double l_Gamyxx = Gamyxx[idx], l_Gamyxy = Gamyxy[idx], l_Gamyxz = Gamyxz[idx];
    const double l_Gamyyy = Gamyyy[idx], l_Gamyyz = Gamyyz[idx], l_Gamyzz = Gamyzz[idx];
    const double l_Gamzxx = Gamzxx[idx], l_Gamzxy = Gamzxy[idx], l_Gamzxz = Gamzxz[idx];
    const double l_Gamzyy = Gamzyy[idx], l_Gamzyz = Gamzyz[idx], l_Gamzzz = Gamzzz[idx];
    double l_Rxy = Rxy[idx], l_Rxz = Rxz[idx], l_Ryz = Ryz[idx];
    double gxxx, gxxy, gxxz, gxyx, gxyy, gxyz, gxzx, gxzy, gxzz;
    double gyyx, gyyy, gyyz, gyzx, gyzy, gyzz, gzzx, gzzy, gzzz;
    // __OFFDIAG_METRIC__
    gxxx = l_gxx * l_Gamxxx + l_gxy * l_Gamyxx + l_gxz * l_Gamzxx;
    gxyx = l_gxx * l_Gamxxy + l_gxy * l_Gamyxy + l_gxz * l_Gamzxy;
    gxzx = l_gxx * l_Gamxxz + l_gxy * l_Gamyxz + l_gxz * l_Gamzxz;
    gyyx = l_gxx * l_Gamxyy + l_gxy * l_Gamyyy + l_gxz * l_Gamzyy;
    gyzx = l_gxx * l_Gamxyz + l_gxy * l_Gamyyz + l_gxz * l_Gamzyz;
    gzzx = l_gxx * l_Gamxzz + l_gxy * l_Gamyzz + l_gxz * l_Gamzzz;

    gxxy = l_gxy * l_Gamxxx + l_gyy * l_Gamyxx + l_gyz * l_Gamzxx;
    gxyy = l_gxy * l_Gamxxy + l_gyy * l_Gamyxy + l_gyz * l_Gamzxy;
    gxzy = l_gxy * l_Gamxxz + l_gyy * l_Gamyxz + l_gyz * l_Gamzxz;
    gyyy = l_gxy * l_Gamxyy + l_gyy * l_Gamyyy + l_gyz * l_Gamzyy;
    gyzy = l_gxy * l_Gamxyz + l_gyy * l_Gamyyz + l_gyz * l_Gamzyz;
    gzzy = l_gxy * l_Gamxzz + l_gyy * l_Gamyzz + l_gyz * l_Gamzzz;

    gxxz = l_gxz * l_Gamxxx + l_gyz * l_Gamyxx + l_gzz * l_Gamzxx;
    gxyz = l_gxz * l_Gamxxy + l_gyz * l_Gamyxy + l_gzz * l_Gamzxy;
    gxzz = l_gxz * l_Gamxxz + l_gyz * l_Gamyxz + l_gzz * l_Gamzxz;
    gyyz = l_gxz * l_Gamxyy + l_gyz * l_Gamyyy + l_gzz * l_Gamzyy;
    gyzz = l_gxz * l_Gamxyz + l_gyz * l_Gamyyz + l_gzz * l_Gamzyz;
    gzzz = l_gxz * l_Gamxzz + l_gyz * l_Gamyzz + l_gzz * l_Gamzzz;
    const double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz +
                         TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    const double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz +
                         TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    const double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz +
                         TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);
    // __OFFDIAG_FORMULAS__
    // Rxy Correction
    l_Rxy = HALF * ( - l_Rxy +
          l_gxx * movx_Res[idx] + l_gxy * Gmx_Res[idx] + l_gxz * Ayy_rhs[idx] +
          l_gxy * ham_Res[idx] + l_gyy * movz_Res[idx] + l_gyz * Gmz_Res[idx] +
          Gamxa * gxyx + Gamya * gyyx + Gamza * gyzx +
          Gamxa * gxxy + Gamya * gxyy + Gamza * gxzy) +
          gupxx * (l_Gamxxx*gxxy + l_Gamyxx*gxyy + l_Gamzxx*gxzy + l_Gamxxy*gxxx + l_Gamyxy*gxyx + l_Gamzxy*gxzx + l_Gamxxx*gxyx + l_Gamyxx*gxyy + l_Gamzxx*gxyz) +
          gupxy * (l_Gamxxx*gxyy + l_Gamyxx*gyyy + l_Gamzxx*gyzy + l_Gamxxy*gxyx + l_Gamyxy*gyyx + l_Gamzxy*gyzx + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz + l_Gamxxy*gxxy + l_Gamyxy*gxyy + l_Gamzxy*gxzy + l_Gamxyy*gxxx + l_Gamyyy*gxyx + l_Gamzyy*gxzx + l_Gamxxx*gyyx + l_Gamyxx*gyyy + l_Gamzxx*gyyz) +
          gupxz * (l_Gamxxx*gxzy + l_Gamyxx*gyzy + l_Gamzxx*gzzy + l_Gamxxy*gxzx + l_Gamyxy*gyzx + l_Gamzxy*gzzx + l_Gamxxz*gxyx + l_Gamyxz*gxyy + l_Gamzxz*gxyz + l_Gamxxz*gxxy + l_Gamyxz*gxyy + l_Gamzxz*gxzy + l_Gamxyz*gxxx + l_Gamyyz*gxyx + l_Gamzyz*gxzx + l_Gamxxx*gyzx + l_Gamyxx*gyzy + l_Gamzxx*gyzz) +
          gupyy * (l_Gamxxy*gxyy + l_Gamyxy*gyyy + l_Gamzxy*gyzy + l_Gamxyy*gxyx + l_Gamyyy*gyyx + l_Gamzyy*gyzx + l_Gamxxy*gyyx + l_Gamyxy*gyyy + l_Gamzxy*gyyz) +
          gupyz * (l_Gamxxy*gxzy + l_Gamyxy*gyzy + l_Gamzxy*gzzy + l_Gamxyy*gxzx + l_Gamyyy*gyzx + l_Gamzyy*gzzx + l_Gamxxz*gyyx + l_Gamyxz*gyyy + l_Gamzxz*gyyz + l_Gamxxz*gxyy + l_Gamyxz*gyyy + l_Gamzxz*gyzy + l_Gamxyz*gxyx + l_Gamyyz*gyyx + l_Gamzyz*gyzx + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) +
          gupzz * (l_Gamxxz*gxzy + l_Gamyxz*gyzy + l_Gamzxz*gzzy + l_Gamxyz*gxzx + l_Gamyyz*gyzx + l_Gamzyz*gzzx + l_Gamxxz*gyzx + l_Gamyxz*gyzy + l_Gamzxz*gyzz);

    // Rxz Correction
    l_Rxz = HALF * ( - l_Rxz +
          l_gxx * movy_Res[idx] + l_gxy * Gmy_Res[idx] + l_gxz * Ayz_rhs[idx] +
          l_gxz * ham_Res[idx] + l_gyz * movz_Res[idx] + l_gzz * Gmz_Res[idx] +
          Gamxa * gxzx + Gamya * gyzx + Gamza * gzzx +
          Gamxa * gxxz + Gamya * gxyz + Gamza * gxzz) +
          gupxx * (l_Gamxxx*gxxz + l_Gamyxx*gxyz + l_Gamzxx*gxzz + l_Gamxxz*gxxx + l_Gamyxz*gxyx + l_Gamzxz*gxzx + l_Gamxxx*gxzx + l_Gamyxx*gxzy + l_Gamzxx*gxzz) +
          gupxy * (l_Gamxxx*gxyz + l_Gamyxx*gyyz + l_Gamzxx*gyzz + l_Gamxxz*gxyx + l_Gamyxz*gyyx + l_Gamzxz*gyzx + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz + l_Gamxxy*gxxz + l_Gamyxy*gxyz + l_Gamzxy*gxzz + l_Gamxyz*gxxx + l_Gamyyz*gxyx + l_Gamzyz*gxzx + l_Gamxxx*gyzx + l_Gamyxx*gyzy + l_Gamzxx*gyzz) +
          gupxz * (l_Gamxxx*gxzz + l_Gamyxx*gyzz + l_Gamzxx*gzzz + l_Gamxxz*gxzx + l_Gamyxz*gyzx + l_Gamzxz*gzzx + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz + l_Gamxxz*gxxz + l_Gamyxz*gxyz + l_Gamzxz*gxzz + l_Gamxzz*gxxx + l_Gamyzz*gxyx + l_Gamzzz*gxzx + l_Gamxxx*gzzx + l_Gamyxx*gzzy + l_Gamzxx*gzzz) +
          gupyy * (l_Gamxxy*gxyz + l_Gamyxy*gyyz + l_Gamzxy*gyzz + l_Gamxyz*gxyx + l_Gamyyz*gyyx + l_Gamzyz*gyzx + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) +
          gupyz * (l_Gamxxy*gxzz + l_Gamyxy*gyzz + l_Gamzxy*gzzz + l_Gamxyz*gxzx + l_Gamyyz*gyzx + l_Gamzyz*gzzx + l_Gamxxz*gyzx + l_Gamyxz*gyzy + l_Gamzxz*gyzz + l_Gamxxz*gxyz + l_Gamyxz*gyyz + l_Gamzxz*gyzz + l_Gamxzz*gxyx + l_Gamyzz*gyyx + l_Gamzzz*gyzx + l_Gamxxy*gzzx + l_Gamyxy*gzzy + l_Gamzxy*gzzz) +
          gupzz * (l_Gamxxz*gxzz + l_Gamyxz*gyzz + l_Gamzxz*gzzz + l_Gamxzz*gxzx + l_Gamyzz*gyzx + l_Gamzzz*gzzx + l_Gamxxz*gzzx + l_Gamyxz*gzzy + l_Gamzxz*gzzz);

    // Ryz Correction
    l_Ryz = HALF * ( - l_Ryz +
          l_gxy * movy_Res[idx] + l_gyy * Gmy_Res[idx] + l_gyz * Ayz_rhs[idx] +
          l_gxz * movx_Res[idx] + l_gyz * Gmx_Res[idx] + l_gzz * Ayy_rhs[idx] +
          Gamxa * gxzy + Gamya * gyzy + Gamza * gzzy +
          Gamxa * gxyz + Gamya * gyyz + Gamza * gyzz) +
          gupxx * (l_Gamxxy*gxxz + l_Gamyxy*gxyz + l_Gamzxy*gxzz + l_Gamxxz*gxxy + l_Gamyxz*gxyy + l_Gamzxz*gxzy + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz) +
          gupxy * (l_Gamxxy*gxyz + l_Gamyxy*gyyz + l_Gamzxy*gyzz + l_Gamxxz*gxyy + l_Gamyxz*gyyy + l_Gamzxz*gyzy + l_Gamxyy*gxzx + l_Gamyyy*gxzy + l_Gamzyy*gxzz + l_Gamxyy*gxxz + l_Gamyyy*gxyz + l_Gamzyy*gxzz + l_Gamxyz*gxxy + l_Gamyyz*gxyy + l_Gamzyz*gxzy + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) +
          gupxz * (l_Gamxxy*gxzz + l_Gamyxy*gyzz + l_Gamzxy*gzzz + l_Gamxxz*gxzy + l_Gamyxz*gyzy + l_Gamzxz*gzzy + l_Gamxyz*gxzx + l_Gamyyz*gxzy + l_Gamzyz*gxzz + l_Gamxyz*gxxz + l_Gamyyz*gxyz + l_Gamzyz*gxzz + l_Gamxzz*gxxy + l_Gamyzz*gxyy + l_Gamzzz*gxzy + l_Gamxxy*gzzx + l_Gamyxy*gzzy + l_Gamzxy*gzzz) +
          gupyy * (l_Gamxyy*gxyz + l_Gamyyy*gyyz + l_Gamzyy*gyzz + l_Gamxyz*gxyy + l_Gamyyz*gyyy + l_Gamzyz*gyzy + l_Gamxyy*gyzx + l_Gamyyy*gyzy + l_Gamzyy*gyzz) +
          gupyz * (l_Gamxyy*gxzz + l_Gamyyy*gyzz + l_Gamzyy*gzzz + l_Gamxyz*gxzy + l_Gamyyz*gyzy + l_Gamzyz*gzzy + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz + l_Gamxyz*gxyz + l_Gamyyz*gyyz + l_Gamzyz*gyzz + l_Gamxzz*gxyy + l_Gamyzz*gyyy + l_Gamzzz*gyzy + l_Gamxyy*gzzx + l_Gamyyy*gzzy + l_Gamzyy*gzzz) +
          gupzz * (l_Gamxyz*gxzz + l_Gamyyz*gyzz + l_Gamzyz*gzzz + l_Gamxzz*gxzy + l_Gamyzz*gyzy + l_Gamzzz*gzzy + l_Gamxyz*gzzx + l_Gamyyz*gzzy + l_Gamzyz*gzzz);
    Rxy[idx] = l_Rxy; Rxz[idx] = l_Rxz; Ryz[idx] = l_Ryz;
}
__global__ void rhs_ricci_a_kernel(
    int ex0, int ex1, int ex2, double T, double* X, double* Y, double* Z,
    double* chi, double* trK,
    double* dxx, double* gxy, double* gxz,
    double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz,
    double* Ayy, double* Ayz, double* Azz,
    double* Gamx, double* Gamy, double* Gamz,
    double* Lap,
    double* betax, double* betay, double* betaz,
    double* dtSfx, double* dtSfy, double* dtSfz,
    double* chi_rhs, double* trK_rhs,
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs,
    double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs,
    double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs,
    double* Lap_rhs,
    double* betax_rhs, double* betay_rhs, double* betaz_rhs,
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
    double* rho, double* Sx, double* Sy, double* Sz,
    double* Sxx, double* Sxy, double* Sxz,
    double* Syy, double* Syz, double* Szz,
    double* Gamxxx, double* Gamxxy, double* Gamxxz,
    double* Gamxyy, double* Gamxyz, double* Gamxzz,
    double* Gamyxx, double* Gamyxy, double* Gamyxz,
    double* Gamyyy, double* Gamyyz, double* Gamyzz,
    double* Gamzxx, double* Gamzxy, double* Gamzxz,
    double* Gamzyy, double* Gamzyz, double* Gamzzz,
    double* Rxx, double* Rxy, double* Rxz,
    double* Ryy, double* Ryz, double* Rzz,
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co
) {
    // ------------------------------------------------------------------------------------
    // bssn_derivatives_kernel
    // ------------------------------------------------------------------------------------

    // 计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2}; // 用于传给 device 函数

    // Load raw state and the geometry-stage scratch values.

    const double l_gxx = dxx[idx] + ONE; const double l_gxy = gxy[idx]; const double l_gxz = gxz[idx];
    const double l_gyy = dyy[idx] + ONE; const double l_gyz = gyz[idx]; const double l_gzz = dzz[idx] + ONE;
    const double l_Axx = Axx[idx]; const double l_Axy = Axy[idx]; const double l_Axz = Axz[idx];
    const double l_Ayy = Ayy[idx]; const double l_Ayz = Ayz[idx]; const double l_Azz = Azz[idx];


    // The six R slots hold inverse metric components until this pass replaces
    // them with physical Ricci components for the constraint pass.
    double gupxx = Rxx[idx], gupxy = Rxy[idx], gupxz = Rxz[idx];
    double gupyy = Ryy[idx], gupyz = Ryz[idx], gupzz = Rzz[idx];

    // The eighteen Gamma slots hold conformal connection coefficients until
    // the physical connection is written back at the end of this pass.
    double l_Gamxxx = Gamxxx[idx], l_Gamxxy = Gamxxy[idx], l_Gamxxz = Gamxxz[idx];
    double l_Gamxyy = Gamxyy[idx], l_Gamxyz = Gamxyz[idx], l_Gamxzz = Gamxzz[idx];
    double l_Gamyxx = Gamyxx[idx], l_Gamyxy = Gamyxy[idx], l_Gamyxz = Gamyxz[idx];
    double l_Gamyyy = Gamyyy[idx], l_Gamyyz = Gamyyz[idx], l_Gamyzz = Gamyzz[idx];
    double l_Gamzxx = Gamzxx[idx], l_Gamzxy = Gamzxy[idx], l_Gamzxz = Gamzxz[idx];
    double l_Gamzyy = Gamzyy[idx], l_Gamzyz = Gamzyz[idx], l_Gamzzz = Gamzzz[idx];


    // ------------------------------------------------------------------------------------
    // bssn_rhs_core_kernel
    // ------------------------------------------------------------------------------------


    // ==========================================
    // Step 1: 初始化 Ricci Tensor (Aij 贡献)
    // ==========================================
    double l_Rxx, l_Rxy, l_Rxz, l_Ryy, l_Ryz, l_Rzz;

    l_Rxx = gupxx * gupxx * l_Axx + gupxy * gupxy * l_Ayy + gupxz * gupxz * l_Azz +
          TWO*(gupxx * gupxy * l_Axy + gupxx * gupxz * l_Axz + gupxy * gupxz * l_Ayz);

    l_Ryy = gupxy * gupxy * l_Axx + gupyy * gupyy * l_Ayy + gupyz * gupyz * l_Azz +
          TWO*(gupxy * gupyy * l_Axy + gupxy * gupyz * l_Axz + gupyy * gupyz * l_Ayz);

    l_Rzz = gupxz * gupxz * l_Axx + gupyz * gupyz * l_Ayy + gupzz * gupzz * l_Azz +
          TWO*(gupxz * gupyz * l_Axy + gupxz * gupzz * l_Axz + gupyz * gupzz * l_Ayz);

    l_Rxy = gupxx * gupxy * l_Axx + gupxy * gupyy * l_Ayy + gupxz * gupyz * l_Azz +
          (gupxx * gupyy + gupxy * gupxy)* l_Axy +
          (gupxx * gupyz + gupxz * gupxy)* l_Axz +
          (gupxy * gupyz + gupxz * gupyy)* l_Ayz;

    l_Rxz = gupxx * gupxz * l_Axx + gupxy * gupyz * l_Ayy + gupxz * gupzz * l_Azz +
          (gupxx * gupyz + gupxy * gupxz)* l_Axy +
          (gupxx * gupzz + gupxz * gupxz)* l_Axz +
          (gupxy * gupzz + gupxz * gupyz)* l_Ayz;

    l_Ryz = gupxy * gupxz * l_Axx + gupyy * gupyz * l_Ayy + gupyz * gupzz * l_Azz +
          (gupxy * gupyz + gupyy * gupxz)* l_Axy +
          (gupxy * gupzz + gupyz * gupxz)* l_Axz +
          (gupyy * gupzz + gupyz * gupyz)* l_Ayz;

    // Preserve inverse metric for the following seed pass and publish Aij Ricci.
    betax_rhs[idx] = gupxx; betay_rhs[idx] = gupxy; betaz_rhs[idx] = gupxz;
    dtSfx_rhs[idx] = gupyy; dtSfy_rhs[idx] = gupyz; dtSfz_rhs[idx] = gupzz;
    Rxx[idx] = l_Rxx; Rxy[idx] = l_Rxy; Rxz[idx] = l_Rxz;
    Ryy[idx] = l_Ryy; Ryz[idx] = l_Ryz; Rzz[idx] = l_Rzz;
}
__global__ void rhs_gamma_derivatives_kernel(RHS_KERNEL_PARAMS) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};
    double Lapx, Lapy, Lapz, Kx, Ky, Kz;
    d_fderivs_point(dims, Lap, &Lapx, &Lapy, &Lapz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, trK, &Kx, &Ky, &Kz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    ham_Res[idx] = Lapx;
    movx_Res[idx] = Lapy;
    movy_Res[idx] = Lapz;
    movz_Res[idx] = Kx;
    Gmx_Res[idx] = Ky;
    Gmy_Res[idx] = Kz;
}

// The Lap/trK derivatives are consumed immediately by all three Gamma seed
// equations. Keep them in registers instead of publishing six intermediate
// values to the global scratch arrays between separate launches.
__global__ void rhs_gamma_seed_fused_kernel(RHS_KERNEL_PARAMS) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};
    double Lapx, Lapy, Lapz, Kx, Ky, Kz;
    d_fderivs_point(dims, Lap, &Lapx, &Lapy, &Lapz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, trK, &Kx, &Ky, &Kz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    // These values are the same scratch values consumed by the old seed
    // kernels. Keeping the aliases explicit preserves the existing equations.
    const double chix = chi_rhs[idx], chiy = trK_rhs[idx], chiz = Lap_rhs[idx];
    const double alpn1 = Lap[idx] + ONE;
    const double chin1 = chi[idx] + ONE;
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];
    const double l_Rxx = Rxx[idx], l_Rxy = Rxy[idx], l_Rxz = Rxz[idx];
    const double l_Ryy = Ryy[idx], l_Ryz = Ryz[idx], l_Rzz = Rzz[idx];
    const double val_Sx = Sx[idx], val_Sy = Sy[idx], val_Sz = Sz[idx];

    const double l_Gamxxx = Gamxxx[idx], l_Gamxxy = Gamxxy[idx], l_Gamxxz = Gamxxz[idx];
    const double l_Gamxyy = Gamxyy[idx], l_Gamxyz = Gamxyz[idx], l_Gamxzz = Gamxzz[idx];
    const double l_Gamyxx = Gamyxx[idx], l_Gamyxy = Gamyxy[idx], l_Gamyxz = Gamyxz[idx];
    const double l_Gamyyy = Gamyyy[idx], l_Gamyyz = Gamyyz[idx], l_Gamyzz = Gamyzz[idx];
    const double l_Gamzxx = Gamzxx[idx], l_Gamzxy = Gamzxy[idx], l_Gamzxz = Gamzxz[idx];
    const double l_Gamzyy = Gamzyy[idx], l_Gamzyz = Gamzyz[idx], l_Gamzzz = Gamzzz[idx];

    const double val_Gamx_rhs = -TWO * (Lapx * l_Rxx + Lapy * l_Rxy + Lapz * l_Rxz) +
        TWO * alpn1 * (
        -F3o2 / chin1 * (chix * l_Rxx + chiy * l_Rxy + chiz * l_Rxz) -
        gupxx * (F2o3 * Kx + EIGHT * PI * val_Sx) -
        gupxy * (F2o3 * Ky + EIGHT * PI * val_Sy) -
        gupxz * (F2o3 * Kz + EIGHT * PI * val_Sz) +
        l_Gamxxx * l_Rxx + l_Gamxyy * l_Ryy + l_Gamxzz * l_Rzz +
        TWO * (l_Gamxxy * l_Rxy + l_Gamxxz * l_Rxz + l_Gamxyz * l_Ryz));
    const double val_Gamy_rhs = -TWO * (Lapx * l_Rxy + Lapy * l_Ryy + Lapz * l_Ryz) +
        TWO * alpn1 * (
        -F3o2 / chin1 * (chix * l_Rxy + chiy * l_Ryy + chiz * l_Ryz) -
        gupxy * (F2o3 * Kx + EIGHT * PI * val_Sx) -
        gupyy * (F2o3 * Ky + EIGHT * PI * val_Sy) -
        gupyz * (F2o3 * Kz + EIGHT * PI * val_Sz) +
        l_Gamyxx * l_Rxx + l_Gamyyy * l_Ryy + l_Gamyzz * l_Rzz +
        TWO * (l_Gamyxy * l_Rxy + l_Gamyxz * l_Rxz + l_Gamyyz * l_Ryz));
    const double val_Gamz_rhs = -TWO * (Lapx * l_Rxz + Lapy * l_Ryz + Lapz * l_Rzz) +
        TWO * alpn1 * (
        -F3o2 / chin1 * (chix * l_Rxz + chiy * l_Ryz + chiz * l_Rzz) -
        gupxz * (F2o3 * Kx + EIGHT * PI * val_Sx) -
        gupyz * (F2o3 * Ky + EIGHT * PI * val_Sy) -
        gupzz * (F2o3 * Kz + EIGHT * PI * val_Sz) +
        l_Gamzxx * l_Rxx + l_Gamzyy * l_Ryy + l_Gamzzz * l_Rzz +
        TWO * (l_Gamzxy * l_Rxy + l_Gamzxz * l_Rxz + l_Gamzyz * l_Ryz));

    Gamx_rhs[idx] = val_Gamx_rhs;
    Gamy_rhs[idx] = val_Gamy_rhs;
    Gamz_rhs[idx] = val_Gamz_rhs;
}

__global__ void rhs_gamy_seed_kernel(
    int ex0, int ex1, int ex2, double T, double* X, double* Y, double* Z,
    double* chi, double* trK,
    double* dxx, double* gxy, double* gxz,
    double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz,
    double* Ayy, double* Ayz, double* Azz,
    double* Gamx, double* Gamy, double* Gamz,
    double* Lap,
    double* betax, double* betay, double* betaz,
    double* dtSfx, double* dtSfy, double* dtSfz,
    double* chi_rhs, double* trK_rhs,
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs,
    double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs,
    double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs,
    double* Lap_rhs,
    double* betax_rhs, double* betay_rhs, double* betaz_rhs,
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
    double* rho, double* Sx, double* Sy, double* Sz,
    double* Sxx, double* Sxy, double* Sxz,
    double* Syy, double* Syz, double* Szz,
    double* Gamxxx, double* Gamxxy, double* Gamxxz,
    double* Gamxyy, double* Gamxyz, double* Gamxzz,
    double* Gamyxx, double* Gamyxy, double* Gamyxz,
    double* Gamyyy, double* Gamyyz, double* Gamyzz,
    double* Gamzxx, double* Gamzxy, double* Gamzxz,
    double* Gamzyy, double* Gamzyz, double* Gamzzz,
    double* Rxx, double* Rxy, double* Rxz,
    double* Ryy, double* Ryz, double* Rzz,
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co
) {
    // ------------------------------------------------------------------------------------
    // bssn_derivatives_kernel
    // ------------------------------------------------------------------------------------

    // 计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);

    // Load raw state and the geometry-stage scratch values.
    const double val_Lap = Lap[idx];
    const double val_chi = chi[idx];
    const double alpn1 = val_Lap + ONE;
    const double chin1 = val_chi + ONE;
    const double chix = chi_rhs[idx], chiy = trK_rhs[idx], chiz = Lap_rhs[idx];



    double gupxy = betay_rhs[idx], gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx];

    double l_Gamyxx = Gamyxx[idx], l_Gamyxy = Gamyxy[idx], l_Gamyxz = Gamyxz[idx];
    double l_Gamyyy = Gamyyy[idx], l_Gamyyz = Gamyyz[idx], l_Gamyzz = Gamyzz[idx];


    // ------------------------------------------------------------------------------------
    // bssn_rhs_core_kernel
    // ------------------------------------------------------------------------------------
    double l_Rxx = Rxx[idx], l_Rxy = Rxy[idx], l_Rxz = Rxz[idx];
    double l_Ryy = Ryy[idx], l_Ryz = Ryz[idx], l_Rzz = Rzz[idx];


    // ==========================================
    // Step 2: 计算 Gam^i_rhs (Part 1: No shift)
    // ==========================================
    const double Lapx = ham_Res[idx], Lapy = movx_Res[idx], Lapz = movy_Res[idx];
    const double Kx = movz_Res[idx], Ky = Gmx_Res[idx], Kz = Gmy_Res[idx];

    // double chix = chix_in[idx]; double chiy = chiy_in[idx]; double chiz = chiz_in[idx];
    double val_Sx = Sx[idx]; double val_Sy = Sy[idx]; double val_Sz = Sz[idx];

    double val_Gamy_rhs = - TWO * (Lapx * l_Rxy + Lapy * l_Ryy + Lapz * l_Ryz) +
        TWO * alpn1 * (
        -F3o2/chin1 * (chix * l_Rxy + chiy * l_Ryy + chiz * l_Ryz) -
        gupxy * (F2o3 * Kx + EIGHT * PI * val_Sx) -
        gupyy * (F2o3 * Ky + EIGHT * PI * val_Sy) -
        gupyz * (F2o3 * Kz + EIGHT * PI * val_Sz) +
        l_Gamyxx * l_Rxx + l_Gamyyy * l_Ryy + l_Gamyzz * l_Rzz +
        TWO * (l_Gamyxy * l_Rxy + l_Gamyxz * l_Rxz + l_Gamyyz * l_Ryz));


    // Publish the seed values for the Ricci/source consumer.
    Gamy_rhs[idx] = val_Gamy_rhs;
}
__global__ void rhs_gamx_seed_kernel(
    int ex0, int ex1, int ex2, double T, double* X, double* Y, double* Z,
    double* chi, double* trK,
    double* dxx, double* gxy, double* gxz,
    double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz,
    double* Ayy, double* Ayz, double* Azz,
    double* Gamx, double* Gamy, double* Gamz,
    double* Lap,
    double* betax, double* betay, double* betaz,
    double* dtSfx, double* dtSfy, double* dtSfz,
    double* chi_rhs, double* trK_rhs,
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs,
    double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs,
    double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs,
    double* Lap_rhs,
    double* betax_rhs, double* betay_rhs, double* betaz_rhs,
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
    double* rho, double* Sx, double* Sy, double* Sz,
    double* Sxx, double* Sxy, double* Sxz,
    double* Syy, double* Syz, double* Szz,
    double* Gamxxx, double* Gamxxy, double* Gamxxz,
    double* Gamxyy, double* Gamxyz, double* Gamxzz,
    double* Gamyxx, double* Gamyxy, double* Gamyxz,
    double* Gamyyy, double* Gamyyz, double* Gamyzz,
    double* Gamzxx, double* Gamzxy, double* Gamzxz,
    double* Gamzyy, double* Gamzyz, double* Gamzzz,
    double* Rxx, double* Rxy, double* Rxz,
    double* Ryy, double* Ryz, double* Rzz,
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co
) {
    // ------------------------------------------------------------------------------------
    // bssn_derivatives_kernel
    // ------------------------------------------------------------------------------------

    // 计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);

    // Load raw state and the geometry-stage scratch values.
    const double val_Lap = Lap[idx];
    const double val_chi = chi[idx];
    const double alpn1 = val_Lap + ONE;
    const double chin1 = val_chi + ONE;
    const double chix = chi_rhs[idx], chiy = trK_rhs[idx], chiz = Lap_rhs[idx];



    double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];

    double l_Gamxxx = Gamxxx[idx], l_Gamxxy = Gamxxy[idx], l_Gamxxz = Gamxxz[idx];
    double l_Gamxyy = Gamxyy[idx], l_Gamxyz = Gamxyz[idx], l_Gamxzz = Gamxzz[idx];


    // ------------------------------------------------------------------------------------
    // bssn_rhs_core_kernel
    // ------------------------------------------------------------------------------------
    double l_Rxx = Rxx[idx], l_Rxy = Rxy[idx], l_Rxz = Rxz[idx];
    double l_Ryy = Ryy[idx], l_Ryz = Ryz[idx], l_Rzz = Rzz[idx];


    // ==========================================
    // Step 2: 计算 Gam^i_rhs (Part 1: No shift)
    // ==========================================
    const double Lapx = ham_Res[idx], Lapy = movx_Res[idx], Lapz = movy_Res[idx];
    const double Kx = movz_Res[idx], Ky = Gmx_Res[idx], Kz = Gmy_Res[idx];

    // double chix = chix_in[idx]; double chiy = chiy_in[idx]; double chiz = chiz_in[idx];
    double val_Sx = Sx[idx]; double val_Sy = Sy[idx]; double val_Sz = Sz[idx];

    // Gamx_rhs
    double val_Gamx_rhs = - TWO * (Lapx * l_Rxx + Lapy * l_Rxy + Lapz * l_Rxz) +
        TWO * alpn1 * (
        -F3o2/chin1 * (chix * l_Rxx + chiy * l_Rxy + chiz * l_Rxz) -
        gupxx * (F2o3 * Kx + EIGHT * PI * val_Sx) -
        gupxy * (F2o3 * Ky + EIGHT * PI * val_Sy) -
        gupxz * (F2o3 * Kz + EIGHT * PI * val_Sz) +
        l_Gamxxx * l_Rxx + l_Gamxyy * l_Ryy + l_Gamxzz * l_Rzz +
        TWO * (l_Gamxxy * l_Rxy + l_Gamxxz * l_Rxz + l_Gamxyz * l_Ryz));
    // Publish the Gamma-x seed and preserve the shared staging values.
    Gamx_rhs[idx] = val_Gamx_rhs;
}
__global__ void rhs_gamz_seed_kernel(
    int ex0, int ex1, int ex2, double T, double* X, double* Y, double* Z,
    double* chi, double* trK,
    double* dxx, double* gxy, double* gxz,
    double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz,
    double* Ayy, double* Ayz, double* Azz,
    double* Gamx, double* Gamy, double* Gamz,
    double* Lap,
    double* betax, double* betay, double* betaz,
    double* dtSfx, double* dtSfy, double* dtSfz,
    double* chi_rhs, double* trK_rhs,
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs,
    double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs,
    double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs,
    double* Lap_rhs,
    double* betax_rhs, double* betay_rhs, double* betaz_rhs,
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
    double* rho, double* Sx, double* Sy, double* Sz,
    double* Sxx, double* Sxy, double* Sxz,
    double* Syy, double* Syz, double* Szz,
    double* Gamxxx, double* Gamxxy, double* Gamxxz,
    double* Gamxyy, double* Gamxyz, double* Gamxzz,
    double* Gamyxx, double* Gamyxy, double* Gamyxz,
    double* Gamyyy, double* Gamyyz, double* Gamyzz,
    double* Gamzxx, double* Gamzxy, double* Gamzxz,
    double* Gamzyy, double* Gamzyz, double* Gamzzz,
    double* Rxx, double* Rxy, double* Rxz,
    double* Ryy, double* Ryz, double* Rzz,
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co
) {
    // ------------------------------------------------------------------------------------
    // bssn_derivatives_kernel
    // ------------------------------------------------------------------------------------

    // 计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);

    // Load raw state and the geometry-stage scratch values.
    const double val_Lap = Lap[idx];
    const double val_chi = chi[idx];
    const double alpn1 = val_Lap + ONE;
    const double chin1 = val_chi + ONE;
    const double chix = chi_rhs[idx], chiy = trK_rhs[idx], chiz = Lap_rhs[idx];



    double gupxz = betaz_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];

    double l_Gamzxx = Gamzxx[idx], l_Gamzxy = Gamzxy[idx], l_Gamzxz = Gamzxz[idx];
    double l_Gamzyy = Gamzyy[idx], l_Gamzyz = Gamzyz[idx], l_Gamzzz = Gamzzz[idx];


    // ------------------------------------------------------------------------------------
    // bssn_rhs_core_kernel
    // ------------------------------------------------------------------------------------
    double l_Rxx = Rxx[idx], l_Rxy = Rxy[idx], l_Rxz = Rxz[idx];
    double l_Ryy = Ryy[idx], l_Ryz = Ryz[idx], l_Rzz = Rzz[idx];


    // ==========================================
    // Step 2: 计算 Gam^i_rhs (Part 1: No shift)
    // ==========================================
    const double Lapx = ham_Res[idx], Lapy = movx_Res[idx], Lapz = movy_Res[idx];
    const double Kx = movz_Res[idx], Ky = Gmx_Res[idx], Kz = Gmy_Res[idx];

    // double chix = chix_in[idx]; double chiy = chiy_in[idx]; double chiz = chiz_in[idx];
    double val_Sx = Sx[idx]; double val_Sy = Sy[idx]; double val_Sz = Sz[idx];

    // Gamx_rhs
    double val_Gamz_rhs = - TWO * (Lapx * l_Rxz + Lapy * l_Ryz + Lapz * l_Rzz) +
        TWO * alpn1 * (
        -F3o2/chin1 * (chix * l_Rxz + chiy * l_Ryz + chiz * l_Rzz) -
        gupxz * (F2o3 * Kx + EIGHT * PI * val_Sx) -
        gupyz * (F2o3 * Ky + EIGHT * PI * val_Sy) -
        gupzz * (F2o3 * Kz + EIGHT * PI * val_Sz) +
        l_Gamzxx * l_Rxx + l_Gamzyy * l_Ryy + l_Gamzzz * l_Rzz +
        TWO * (l_Gamzxy * l_Rxy + l_Gamzxz * l_Rxz + l_Gamzyz * l_Ryz));

    // Publish the seed values for the Ricci/source consumer.
    // Publish the Gamma-z seed and preserve the shared staging values.
    Gamz_rhs[idx] = val_Gamz_rhs;
}
__global__ void rhs_geometry_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)trK;
    (void)Axx; (void)Axy; (void)Axz; (void)Ayy; (void)Ayz; (void)Azz;
    (void)Gamx; (void)Gamy; (void)Gamz; (void)Lap;
    (void)dtSfx; (void)dtSfy; (void)dtSfz;
    (void)chi_rhs; (void)trK_rhs;
    (void)Sxx; (void)Sxy; (void)Sxz; (void)Syy; (void)Syz; (void)Szz;
    (void)rho; (void)Sx; (void)Sy; (void)Sz;
    (void)dtSfx_rhs; (void)dtSfy_rhs; (void)dtSfz_rhs;
    (void)Lap_rhs; (void)ham_Res; (void)movx_Res; (void)movy_Res; (void)movz_Res;
    (void)Gmx_Res; (void)Gmy_Res; (void)Gmz_Res; (void)eps; (void)co;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};

    const double l_gxx = dxx[idx] + ONE, l_gxy = gxy[idx], l_gxz = gxz[idx];
    const double l_gyy = dyy[idx] + ONE, l_gyz = gyz[idx], l_gzz = dzz[idx] + ONE;

    double betaxx, betaxy, betaxz, betayx, betayy, betayz, betazx, betazy, betazz;
    d_fderivs_point(dims, betax, &betaxx, &betaxy, &betaxz,
                    X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betay, &betayx, &betayy, &betayz,
                    X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betaz, &betazx, &betazy, &betazz,
                    X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

    double chix, chiy, chiz;
    d_fderivs_point(dims, chi, &chix, &chiy, &chiz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    double gxxx, gxxy, gxxz, gxyx, gxyy, gxyz, gxzx, gxzy, gxzz;
    double gyyx, gyyy, gyyz, gyzx, gyzy, gyzz, gzzx, gzzy, gzzz;
    d_fderivs_point(dims, dxx, &gxxx, &gxxy, &gxxz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gxy, &gxyx, &gxyy, &gxyz,
                    X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gxz, &gxzx, &gxzy, &gxzz,
                    X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, dyy, &gyyx, &gyyy, &gyyz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gyz, &gyzx, &gyzy, &gyzz,
                    X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, dzz, &gzzx, &gzzy, &gzzz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    const double detg = l_gxx * (l_gyy * l_gzz - l_gyz * l_gyz)
                      - l_gxy * (l_gxy * l_gzz - l_gyz * l_gxz)
                      + l_gxz * (l_gxy * l_gyz - l_gyy * l_gxz);
    const double gupxx = (l_gyy * l_gzz - l_gyz * l_gyz) / detg;
    const double gupxy = -(l_gxy * l_gzz - l_gyz * l_gxz) / detg;
    const double gupxz = (l_gxy * l_gyz - l_gyy * l_gxz) / detg;
    const double gupyy = (l_gxx * l_gzz - l_gxz * l_gxz) / detg;
    const double gupyz = -(l_gxx * l_gyz - l_gxy * l_gxz) / detg;
    const double gupzz = (l_gxx * l_gyy - l_gxy * l_gxy) / detg;

    const double l_Gamxxx = HALF * (gupxx * gxxx + gupxy * (TWO * gxyx - gxxy) + gupxz * (TWO * gxzx - gxxz));
    const double l_Gamyxx = HALF * (gupxy * gxxx + gupyy * (TWO * gxyx - gxxy) + gupyz * (TWO * gxzx - gxxz));
    const double l_Gamzxx = HALF * (gupxz * gxxx + gupyz * (TWO * gxyx - gxxy) + gupzz * (TWO * gxzx - gxxz));
    const double l_Gamxyy = HALF * (gupxx * (TWO * gxyy - gyyx) + gupxy * gyyy + gupxz * (TWO * gyzy - gyyz));
    const double l_Gamyyy = HALF * (gupxy * (TWO * gxyy - gyyx) + gupyy * gyyy + gupyz * (TWO * gyzy - gyyz));
    const double l_Gamzyy = HALF * (gupxz * (TWO * gxyy - gyyx) + gupyz * gyyy + gupzz * (TWO * gyzy - gyyz));
    const double l_Gamxzz = HALF * (gupxx * (TWO * gxzz - gzzx) + gupxy * (TWO * gyzz - gzzy) + gupxz * gzzz);
    const double l_Gamyzz = HALF * (gupxy * (TWO * gxzz - gzzx) + gupyy * (TWO * gyzz - gzzy) + gupyz * gzzz);
    const double l_Gamzzz = HALF * (gupxz * (TWO * gxzz - gzzx) + gupyz * (TWO * gyzz - gzzy) + gupzz * gzzz);
    const double l_Gamxxy = HALF * (gupxx * gxxy + gupxy * gyyx + gupxz * (gxzy + gyzx - gxyz));
    const double l_Gamyxy = HALF * (gupxy * gxxy + gupyy * gyyx + gupyz * (gxzy + gyzx - gxyz));
    const double l_Gamzxy = HALF * (gupxz * gxxy + gupyz * gyyx + gupzz * (gxzy + gyzx - gxyz));
    const double l_Gamxxz = HALF * (gupxx * gxxz + gupxy * (gxyz + gyzx - gxzy) + gupxz * gzzx);
    const double l_Gamyxz = HALF * (gupxy * gxxz + gupyy * (gxyz + gyzx - gxzy) + gupyz * gzzx);
    const double l_Gamzxz = HALF * (gupxz * gxxz + gupyz * (gxyz + gyzx - gxzy) + gupzz * gzzx);
    const double l_Gamxyz = HALF * (gupxx * (gxyz + gxzy - gyzx) + gupxy * gyyz + gupxz * gzzy);
    const double l_Gamyyz = HALF * (gupxy * (gxyz + gxzy - gyzx) + gupyy * gyyz + gupyz * gzzy);
    const double l_Gamzyz = HALF * (gupxz * (gxyz + gxzy - gyzx) + gupyz * gyyz + gupzz * gzzy);

    // Existing arrays carry producer values only until the consumer runs.
    gxx_rhs[idx] = betaxx; gxy_rhs[idx] = betaxy; gxz_rhs[idx] = betaxz;
    gyy_rhs[idx] = betayx; gyz_rhs[idx] = betayy; gzz_rhs[idx] = betayz;
    Axx_rhs[idx] = betazx; Axy_rhs[idx] = betazy; Axz_rhs[idx] = betazz;
    chi_rhs[idx] = chix; trK_rhs[idx] = chiy; Lap_rhs[idx] = chiz;
    Gamxxx[idx] = l_Gamxxx; Gamxxy[idx] = l_Gamxxy; Gamxxz[idx] = l_Gamxxz;
    Gamxyy[idx] = l_Gamxyy; Gamxyz[idx] = l_Gamxyz; Gamxzz[idx] = l_Gamxzz;
    Gamyxx[idx] = l_Gamyxx; Gamyxy[idx] = l_Gamyxy; Gamyxz[idx] = l_Gamyxz;
    Gamyyy[idx] = l_Gamyyy; Gamyyz[idx] = l_Gamyyz; Gamyzz[idx] = l_Gamyzz;
    Gamzxx[idx] = l_Gamzxx; Gamzxy[idx] = l_Gamzxy; Gamzxz[idx] = l_Gamzxz;
    Gamzyy[idx] = l_Gamzyy; Gamzyz[idx] = l_Gamzyz; Gamzzz[idx] = l_Gamzzz;
    Rxx[idx] = gupxx; Rxy[idx] = gupxy; Rxz[idx] = gupxz;
    Ryy[idx] = gupyy; Ryz[idx] = gupyz; Rzz[idx] = gupzz;
}
__global__ void rhs_source_metric_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const double val_Lap = Lap[idx];
    const double val_chi = chi[idx];
    const double alpn1 = val_Lap + ONE;
    const double chin1 = val_chi + ONE;
    const double val_trK = trK[idx];
    const double l_gxx = dxx[idx] + ONE, l_gxy = gxy[idx], l_gxz = gxz[idx];
    const double l_gyy = dyy[idx] + ONE, l_gyz = gyz[idx], l_gzz = dzz[idx] + ONE;
    const double l_Axx = Axx[idx], l_Axy = Axy[idx], l_Axz = Axz[idx];
    const double l_Ayy = Ayy[idx], l_Ayz = Ayz[idx], l_Azz = Azz[idx];

    // Geometry derivatives remain in metric RHS slots until this pass.
    const double betaxx = gxx_rhs[idx], betaxy = gxy_rhs[idx], betaxz = gxz_rhs[idx];
    const double betayx = gyy_rhs[idx], betayy = gyz_rhs[idx], betayz = gzz_rhs[idx];
    const double betazx = Axx_rhs[idx], betazy = Axy_rhs[idx], betazz = Axz_rhs[idx];
    const double div_beta = betaxx + betayy + betazz;
    // Scalar and conformal metric RHS terms use the beta derivatives staged above.
    chi_rhs[idx] = F2o3 * chin1 * (alpn1 * val_trK - div_beta);
    gxx_rhs[idx] = -TWO * alpn1 * l_Axx - F2o3 * l_gxx * div_beta +
                   TWO * (l_gxx * betaxx + l_gxy * betayx + l_gxz * betazx);
    gyy_rhs[idx] = -TWO * alpn1 * l_Ayy - F2o3 * l_gyy * div_beta +
                   TWO * (l_gxy * betaxy + l_gyy * betayy + l_gyz * betazy);
    gzz_rhs[idx] = -TWO * alpn1 * l_Azz - F2o3 * l_gzz * div_beta +
                   TWO * (l_gxz * betaxz + l_gyz * betayz + l_gzz * betazz);
    gxy_rhs[idx] = -TWO * alpn1 * l_Axy + F1o3 * l_gxy * div_beta +
                   l_gxx * betaxy + l_gxz * betazy + l_gyy * betayx + l_gyz * betazx - l_gxy * betazz;
    gyz_rhs[idx] = -TWO * alpn1 * l_Ayz + F1o3 * l_gyz * div_beta +
                   l_gxy * betaxz + l_gyy * betayz + l_gxz * betaxy + l_gzz * betazy - l_gyz * betaxx;
    gxz_rhs[idx] = -TWO * alpn1 * l_Axz + F1o3 * l_gxz * div_beta +
                   l_gxx * betaxz + l_gxy * betayz + l_gyz * betayx + l_gzz * betazx - l_gxz * betayy;
}
__global__ void rhs_source_chi_hessian_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)Gmz_Res; (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};
    double chix, chiy, chiz;
    d_fderivs_point(dims, chi, &chix, &chiy, &chiz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    double fxx, fxy, fxz, fyy, fyz, fzz;
    d_fdderivs_point(dims, chi, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz,
                     X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    fxx -= Gamxxx[idx] * chix + Gamyxx[idx] * chiy + Gamzxx[idx] * chiz;
    fxy -= Gamxxy[idx] * chix + Gamyxy[idx] * chiy + Gamzxy[idx] * chiz;
    fxz -= Gamxxz[idx] * chix + Gamyxz[idx] * chiy + Gamzxz[idx] * chiz;
    fyy -= Gamxyy[idx] * chix + Gamyyy[idx] * chiy + Gamzyy[idx] * chiz;
    fyz -= Gamxyz[idx] * chix + Gamyyz[idx] * chiy + Gamzyz[idx] * chiz;
    fzz -= Gamxzz[idx] * chix + Gamyzz[idx] * chiy + Gamzzz[idx] * chiz;
    ham_Res[idx] = fxx; movx_Res[idx] = fxy; movy_Res[idx] = fxz;
    movz_Res[idx] = fyy; Gmx_Res[idx] = fyz; Gmy_Res[idx] = fzz;
}
__global__ void rhs_source_chi_ricci_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)Gmz_Res; (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};
    const double chin1 = chi[idx] + ONE;
    const double l_gxx = dxx[idx] + ONE, l_gxy = gxy[idx], l_gxz = gxz[idx];
    const double l_gyy = dyy[idx] + ONE, l_gyz = gyz[idx], l_gzz = dzz[idx] + ONE;
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];
    double chix, chiy, chiz;
    if (symmetry == 1) {
        chix = chi_rhs[idx];
        chiy = trK_rhs[idx];
        chiz = Lap_rhs[idx];
    } else {
        d_fderivs_point(dims, chi, &chix, &chiy, &chiz,
                        X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    }
    const double fxx = ham_Res[idx], fxy = movx_Res[idx], fxz = movy_Res[idx];
    const double fyy = movz_Res[idx], fyz = Gmx_Res[idx], fzz = Gmy_Res[idx];
    const double f_scalar = gupxx * (fxx - F3o2/chin1 * chix * chix) +
                            gupyy * (fyy - F3o2/chin1 * chiy * chiy) +
                            gupzz * (fzz - F3o2/chin1 * chiz * chiz) +
                            TWO * (gupxy * (fxy - F3o2/chin1 * chix * chiy) +
                                   gupxz * (fxz - F3o2/chin1 * chix * chiz) +
                                   gupyz * (fyz - F3o2/chin1 * chiy * chiz));
    Rxx[idx] += (fxx - chix*chix/chin1/TWO + l_gxx * f_scalar)/chin1/TWO;
    Ryy[idx] += (fyy - chiy*chiy/chin1/TWO + l_gyy * f_scalar)/chin1/TWO;
    Rzz[idx] += (fzz - chiz*chiz/chin1/TWO + l_gzz * f_scalar)/chin1/TWO;
    Rxy[idx] += (fxy - chix*chiy/chin1/TWO + l_gxy * f_scalar)/chin1/TWO;
    Rxz[idx] += (fxz - chix*chiz/chin1/TWO + l_gxz * f_scalar)/chin1/TWO;
    Ryz[idx] += (fyz - chiy*chiz/chin1/TWO + l_gyz * f_scalar)/chin1/TWO;
}
__global__ void rhs_source_physical_gamma_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};
    const double chin1 = chi[idx] + ONE;
    const double l_gxx = dxx[idx] + ONE, l_gxy = gxy[idx], l_gxz = gxz[idx];
    const double l_gyy = dyy[idx] + ONE, l_gyz = gyz[idx], l_gzz = dzz[idx] + ONE;
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];
    double chix, chiy, chiz;
    if (symmetry == 1) {
        chix = chi_rhs[idx];
        chiy = trK_rhs[idx];
        chiz = Lap_rhs[idx];
    } else {
        d_fderivs_point(dims, chi, &chix, &chiy, &chiz,
                        X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    }
    const double gx_phy = (gupxx * chix + gupxy * chiy + gupxz * chiz)/chin1;
    const double gy_phy = (gupxy * chix + gupyy * chiy + gupyz * chiz)/chin1;
    const double gz_phy = (gupxz * chix + gupyz * chiy + gupzz * chiz)/chin1;
    Gamxxx[idx] -= ((chix + chix)/chin1 - l_gxx * gx_phy)*HALF;
    Gamyxx[idx] -= (                            - l_gxx * gy_phy)*HALF;
    Gamzxx[idx] -= (                            - l_gxx * gz_phy)*HALF;
    Gamxyy[idx] -= (                            - l_gyy * gx_phy)*HALF;
    Gamyyy[idx] -= ((chiy + chiy)/chin1 - l_gyy * gy_phy)*HALF;
    Gamzyy[idx] -= (                            - l_gyy * gz_phy)*HALF;
    Gamxzz[idx] -= (                            - l_gzz * gx_phy)*HALF;
    Gamyzz[idx] -= (                            - l_gzz * gy_phy)*HALF;
    Gamzzz[idx] -= ((chiz + chiz)/chin1 - l_gzz * gz_phy)*HALF;
    Gamxxy[idx] -= (chiy/chin1 - l_gxy * gx_phy)*HALF;
    Gamyxy[idx] -= (chix/chin1 - l_gxy * gy_phy)*HALF;
    Gamzxy[idx] -= (                 - l_gxy * gz_phy)*HALF;
    Gamxxz[idx] -= (chiz/chin1 - l_gxz * gx_phy)*HALF;
    Gamyxz[idx] -= (                 - l_gxz * gy_phy)*HALF;
    Gamzxz[idx] -= (chix/chin1 - l_gxz * gz_phy)*HALF;
    Gamxyz[idx] -= (                 - l_gyz * gx_phy)*HALF;
    Gamyyz[idx] -= (chiz/chin1 - l_gyz * gy_phy)*HALF;
    Gamzyz[idx] -= (chiy/chin1 - l_gyz * gz_phy)*HALF;
}
__global__ void rhs_source_lapse_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];
    double Lapx, Lapy, Lapz;
    d_fderivs_point(dims, Lap, &Lapx, &Lapy, &Lapz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    double fxx, fxy, fxz, fyy, fyz, fzz;
    d_fdderivs_point(dims, Lap, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz,
                     X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    fxx -= Gamxxx[idx]*Lapx + Gamyxx[idx]*Lapy + Gamzxx[idx]*Lapz;
    fyy -= Gamxyy[idx]*Lapx + Gamyyy[idx]*Lapy + Gamzyy[idx]*Lapz;
    fzz -= Gamxzz[idx]*Lapx + Gamyzz[idx]*Lapy + Gamzzz[idx]*Lapz;
    fxy -= Gamxxy[idx]*Lapx + Gamyxy[idx]*Lapy + Gamzxy[idx]*Lapz;
    fxz -= Gamxxz[idx]*Lapx + Gamyxz[idx]*Lapy + Gamzxz[idx]*Lapz;
    fyz -= Gamxyz[idx]*Lapx + Gamyyz[idx]*Lapy + Gamzyz[idx]*Lapz;
    const double trK_rhs_val = gupxx * fxx + gupyy * fyy + gupzz * fzz +
                               TWO * (gupxy * fxy + gupxz * fxz + gupyz * fyz);
    ham_Res[idx] = fxx; movx_Res[idx] = fxy; movy_Res[idx] = fxz;
    movz_Res[idx] = fyy; Gmx_Res[idx] = fyz; Gmy_Res[idx] = fzz;
    Gmz_Res[idx] = trK_rhs_val;
}
__global__ void rhs_source_trace_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const double alpn1 = Lap[idx] + ONE;
    const double chin1 = chi[idx] + ONE;
    const double val_trK = trK[idx];
    const double l_Axx = Axx[idx], l_Axy = Axy[idx], l_Axz = Axz[idx];
    const double l_Ayy = Ayy[idx], l_Ayz = Ayz[idx], l_Azz = Azz[idx];
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];
    const double term_xx = gupxx*l_Axx*l_Axx + gupyy*l_Axy*l_Axy + gupzz*l_Axz*l_Axz + TWO*(gupxy*l_Axx*l_Axy + gupxz*l_Axx*l_Axz + gupyz*l_Axy*l_Axz);
    const double term_yy = gupxx*l_Axy*l_Axy + gupyy*l_Ayy*l_Ayy + gupzz*l_Ayz*l_Ayz + TWO*(gupxy*l_Axy*l_Ayy + gupxz*l_Axy*l_Ayz + gupyz*l_Ayy*l_Ayz);
    const double term_zz = gupxx*l_Axz*l_Axz + gupyy*l_Ayz*l_Ayz + gupzz*l_Azz*l_Azz + TWO*(gupxy*l_Axz*l_Ayz + gupxz*l_Axz*l_Azz + gupyz*l_Ayz*l_Azz);
    const double term_xy = gupxx*l_Axx*l_Axy + gupyy*l_Axy*l_Ayy + gupzz*l_Axz*l_Ayz + gupxy*(l_Axx*l_Ayy + l_Axy*l_Axy) + gupxz*(l_Axx*l_Ayz + l_Axz*l_Axy) + gupyz*(l_Axy*l_Ayz + l_Axz*l_Ayy);
    const double term_xz = gupxx*l_Axx*l_Axz + gupyy*l_Axy*l_Ayz + gupzz*l_Axz*l_Azz + gupxy*(l_Axx*l_Ayz + l_Axy*l_Axz) + gupxz*(l_Axx*l_Azz + l_Axz*l_Axz) + gupyz*(l_Axy*l_Azz + l_Axz*l_Ayz);
    const double term_yz = gupxx*l_Axy*l_Axz + gupyy*l_Ayy*l_Ayz + gupzz*l_Ayz*l_Azz + gupxy*(l_Axy*l_Ayz + l_Ayy*l_Axz) + gupxz*(l_Axy*l_Azz + l_Ayz*l_Axz) + gupyz*(l_Ayy*l_Azz + l_Ayz*l_Ayz);
    const double trA2 = gupxx*term_xx + gupyy*term_yy + gupzz*term_zz + TWO*(gupxy*term_xy + gupxz*term_xz + gupyz*term_yz);
    const double S = chin1 * (gupxx*Sxx[idx] + gupyy*Syy[idx] + gupzz*Szz[idx] + TWO*(gupxy*Sxy[idx] + gupxz*Sxz[idx] + gupyz*Syz[idx]));
    // The lapse contraction is consumed here; reuse its slot for f_trace until both Aij kernels finish.
    const double trK_hessian = Gmz_Res[idx];
    const double f = F2o3*val_trK*val_trK - trA2 - F16*PI*rho[idx] + EIGHT*PI*S;
    Gmz_Res[idx] = -F1o3 * (trK_hessian + alpn1/chin1 * f);
    trK_rhs[idx] = -chin1*trK_hessian + alpn1*(F1o3*val_trK*val_trK + trA2 + FOUR*PI*(rho[idx] + S));
}

__global__ void rhs_source_a_diag_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};
    const double alpn1 = Lap[idx] + ONE, chin1 = chi[idx] + ONE, val_trK = trK[idx];
    const double l_gxx = dxx[idx] + ONE, l_gyy = dyy[idx] + ONE, l_gzz = dzz[idx] + ONE;
    const double l_Axx = Axx[idx], l_Axy = Axy[idx], l_Axz = Axz[idx];
    const double l_Ayy = Ayy[idx], l_Ayz = Ayz[idx], l_Azz = Azz[idx];
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];
    double betaxx, betaxy, betaxz, betayx, betayy, betayz, betazx, betazy, betazz;
    d_fderivs_point(dims, betax, &betaxx, &betaxy, &betaxz, X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betay, &betayx, &betayy, &betayz, X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betaz, &betazx, &betazy, &betazz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);
    const double div_beta = betaxx + betayy + betazz;
    const double term_xx = gupxx*l_Axx*l_Axx + gupyy*l_Axy*l_Axy + gupzz*l_Axz*l_Axz + TWO*(gupxy*l_Axx*l_Axy + gupxz*l_Axx*l_Axz + gupyz*l_Axy*l_Axz);
    const double term_yy = gupxx*l_Axy*l_Axy + gupyy*l_Ayy*l_Ayy + gupzz*l_Ayz*l_Ayz + TWO*(gupxy*l_Axy*l_Ayy + gupxz*l_Axy*l_Ayz + gupyz*l_Ayy*l_Ayz);
    const double term_zz = gupxx*l_Axz*l_Axz + gupyy*l_Ayz*l_Ayz + gupzz*l_Azz*l_Azz + TWO*(gupxy*l_Axz*l_Ayz + gupxz*l_Axz*l_Azz + gupyz*l_Ayz*l_Azz);
    const double f_trace = Gmz_Res[idx];
    const double src_xx = alpn1*(Rxx[idx] - EIGHT*PI*Sxx[idx]) - ham_Res[idx] - l_gxx*f_trace;
    const double src_yy = alpn1*(Ryy[idx] - EIGHT*PI*Syy[idx]) - movz_Res[idx] - l_gyy*f_trace;
    const double src_zz = alpn1*(Rzz[idx] - EIGHT*PI*Szz[idx]) - Gmy_Res[idx] - l_gzz*f_trace;
    Axx_rhs[idx] = chin1*src_xx + alpn1*(val_trK*l_Axx - TWO*term_xx) + TWO*(l_Axx*betaxx + l_Axy*betayx + l_Axz*betazx) - F2o3*l_Axx*div_beta;
    Ayy_rhs[idx] = chin1*src_yy + alpn1*(val_trK*l_Ayy - TWO*term_yy) + TWO*(l_Axy*betaxy + l_Ayy*betayy + l_Ayz*betazy) - F2o3*l_Ayy*div_beta;
    Azz_rhs[idx] = chin1*src_zz + alpn1*(val_trK*l_Azz - TWO*term_zz) + TWO*(l_Axz*betaxz + l_Ayz*betayz + l_Azz*betazz) - F2o3*l_Azz*div_beta;
}

__global__ void rhs_source_a_offdiag_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};
    const double alpn1 = Lap[idx] + ONE, chin1 = chi[idx] + ONE, val_trK = trK[idx];
    const double l_gxy = gxy[idx], l_gxz = gxz[idx], l_gyz = gyz[idx];
    const double l_Axx = Axx[idx], l_Axy = Axy[idx], l_Axz = Axz[idx];
    const double l_Ayy = Ayy[idx], l_Ayz = Ayz[idx], l_Azz = Azz[idx];
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];
    double betaxx, betaxy, betaxz, betayx, betayy, betayz, betazx, betazy, betazz;
    d_fderivs_point(dims, betax, &betaxx, &betaxy, &betaxz, X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betay, &betayx, &betayy, &betayz, X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betaz, &betazx, &betazy, &betazz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);
    const double div_beta = betaxx + betayy + betazz;
    const double term_xy = gupxx*l_Axx*l_Axy + gupyy*l_Axy*l_Ayy + gupzz*l_Axz*l_Ayz + gupxy*(l_Axx*l_Ayy + l_Axy*l_Axy) + gupxz*(l_Axx*l_Ayz + l_Axz*l_Axy) + gupyz*(l_Axy*l_Ayz + l_Axz*l_Ayy);
    const double term_xz = gupxx*l_Axx*l_Axz + gupyy*l_Axy*l_Ayz + gupzz*l_Axz*l_Azz + gupxy*(l_Axx*l_Ayz + l_Axy*l_Axz) + gupxz*(l_Axx*l_Azz + l_Axz*l_Axz) + gupyz*(l_Axy*l_Azz + l_Axz*l_Ayz);
    const double term_yz = gupxx*l_Axy*l_Axz + gupyy*l_Ayy*l_Ayz + gupzz*l_Ayz*l_Azz + gupxy*(l_Axy*l_Ayz + l_Ayy*l_Axz) + gupxz*(l_Axy*l_Azz + l_Ayz*l_Axz) + gupyz*(l_Ayy*l_Azz + l_Ayz*l_Ayz);
    const double f_trace = Gmz_Res[idx];
    const double src_xy = alpn1*(Rxy[idx] - EIGHT*PI*Sxy[idx]) - movx_Res[idx] - l_gxy*f_trace;
    const double src_xz = alpn1*(Rxz[idx] - EIGHT*PI*Sxz[idx]) - movy_Res[idx] - l_gxz*f_trace;
    const double src_yz = alpn1*(Ryz[idx] - EIGHT*PI*Syz[idx]) - Gmx_Res[idx] - l_gyz*f_trace;
    Axy_rhs[idx] = chin1*src_xy + alpn1*(val_trK*l_Axy - TWO*term_xy) + l_Axx*betaxy + l_Axz*betazy + l_Ayy*betayx + l_Ayz*betazx - l_Axy*betazz + F1o3*l_Axy*div_beta;
    Ayz_rhs[idx] = chin1*src_yz + alpn1*(val_trK*l_Ayz - TWO*term_yz) + l_Axy*betaxz + l_Ayy*betayz + l_Axz*betaxy + l_Azz*betazy - l_Ayz*betaxx + F1o3*l_Ayz*div_beta;
    Axz_rhs[idx] = chin1*src_xz + alpn1*(val_trK*l_Axz - TWO*term_xz) + l_Axx*betaxz + l_Axy*betayz + l_Ayz*betayx + l_Azz*betazx - l_Axz*betayy + F1o3*l_Axz*div_beta;
}

__global__ void rhs_source_metric_a_equatorial_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const double alpn1 = Lap[idx] + ONE, chin1 = chi[idx] + ONE, val_trK = trK[idx];
    const double l_gxx = dxx[idx] + ONE, l_gxy = gxy[idx], l_gxz = gxz[idx];
    const double l_gyy = dyy[idx] + ONE, l_gyz = gyz[idx], l_gzz = dzz[idx] + ONE;
    const double l_Axx = Axx[idx], l_Axy = Axy[idx], l_Axz = Axz[idx];
    const double l_Ayy = Ayy[idx], l_Ayz = Ayz[idx], l_Azz = Azz[idx];
    const double gupxx = betax_rhs[idx], gupxy = betay_rhs[idx], gupxz = betaz_rhs[idx];
    const double gupyy = dtSfx_rhs[idx], gupyz = dtSfy_rhs[idx], gupzz = dtSfz_rhs[idx];

    // Geometry publishes these nine values in RHS scratch slots. Load every
    // value before any fused output overwrites one of those slots.
    const double betaxx = gxx_rhs[idx], betaxy = gxy_rhs[idx], betaxz = gxz_rhs[idx];
    const double betayx = gyy_rhs[idx], betayy = gyz_rhs[idx], betayz = gzz_rhs[idx];
    const double betazx = Axx_rhs[idx], betazy = Axy_rhs[idx], betazz = Axz_rhs[idx];
    const double div_beta = betaxx + betayy + betazz;

    const double term_xx = gupxx*l_Axx*l_Axx + gupyy*l_Axy*l_Axy + gupzz*l_Axz*l_Axz + TWO*(gupxy*l_Axx*l_Axy + gupxz*l_Axx*l_Axz + gupyz*l_Axy*l_Axz);
    const double term_yy = gupxx*l_Axy*l_Axy + gupyy*l_Ayy*l_Ayy + gupzz*l_Ayz*l_Ayz + TWO*(gupxy*l_Axy*l_Ayy + gupxz*l_Axy*l_Ayz + gupyz*l_Ayy*l_Ayz);
    const double term_zz = gupxx*l_Axz*l_Axz + gupyy*l_Ayz*l_Ayz + gupzz*l_Azz*l_Azz + TWO*(gupxy*l_Axz*l_Ayz + gupxz*l_Axz*l_Azz + gupyz*l_Ayz*l_Azz);
    const double term_xy = gupxx*l_Axx*l_Axy + gupyy*l_Axy*l_Ayy + gupzz*l_Axz*l_Ayz + gupxy*(l_Axx*l_Ayy + l_Axy*l_Axy) + gupxz*(l_Axx*l_Ayz + l_Axz*l_Axy) + gupyz*(l_Axy*l_Ayz + l_Axz*l_Ayy);
    const double term_xz = gupxx*l_Axx*l_Axz + gupyy*l_Axy*l_Ayz + gupzz*l_Axz*l_Azz + gupxy*(l_Axx*l_Ayz + l_Axy*l_Axz) + gupxz*(l_Axx*l_Azz + l_Axz*l_Axz) + gupyz*(l_Axy*l_Azz + l_Axz*l_Ayz);
    const double term_yz = gupxx*l_Axy*l_Axz + gupyy*l_Ayy*l_Ayz + gupzz*l_Ayz*l_Azz + gupxy*(l_Axy*l_Ayz + l_Ayy*l_Axz) + gupxz*(l_Axy*l_Azz + l_Ayz*l_Axz) + gupyz*(l_Ayy*l_Azz + l_Ayz*l_Ayz);
    const double f_trace = Gmz_Res[idx];
    const double src_xx = alpn1*(Rxx[idx] - EIGHT*PI*Sxx[idx]) - ham_Res[idx] - l_gxx*f_trace;
    const double src_yy = alpn1*(Ryy[idx] - EIGHT*PI*Syy[idx]) - movz_Res[idx] - l_gyy*f_trace;
    const double src_zz = alpn1*(Rzz[idx] - EIGHT*PI*Szz[idx]) - Gmy_Res[idx] - l_gzz*f_trace;
    const double src_xy = alpn1*(Rxy[idx] - EIGHT*PI*Sxy[idx]) - movx_Res[idx] - l_gxy*f_trace;
    const double src_xz = alpn1*(Rxz[idx] - EIGHT*PI*Sxz[idx]) - movy_Res[idx] - l_gxz*f_trace;
    const double src_yz = alpn1*(Ryz[idx] - EIGHT*PI*Syz[idx]) - Gmx_Res[idx] - l_gyz*f_trace;

    chi_rhs[idx] = F2o3 * chin1 * (alpn1 * val_trK - div_beta);
    gxx_rhs[idx] = -TWO * alpn1 * l_Axx - F2o3 * l_gxx * div_beta +
                   TWO * (l_gxx * betaxx + l_gxy * betayx + l_gxz * betazx);
    gyy_rhs[idx] = -TWO * alpn1 * l_Ayy - F2o3 * l_gyy * div_beta +
                   TWO * (l_gxy * betaxy + l_gyy * betayy + l_gyz * betazy);
    gzz_rhs[idx] = -TWO * alpn1 * l_Azz - F2o3 * l_gzz * div_beta +
                   TWO * (l_gxz * betaxz + l_gyz * betayz + l_gzz * betazz);
    gxy_rhs[idx] = -TWO * alpn1 * l_Axy + F1o3 * l_gxy * div_beta +
                   l_gxx * betaxy + l_gxz * betazy + l_gyy * betayx + l_gyz * betazx - l_gxy * betazz;
    gyz_rhs[idx] = -TWO * alpn1 * l_Ayz + F1o3 * l_gyz * div_beta +
                   l_gxy * betaxz + l_gyy * betayz + l_gxz * betaxy + l_gzz * betazy - l_gyz * betaxx;
    gxz_rhs[idx] = -TWO * alpn1 * l_Axz + F1o3 * l_gxz * div_beta +
                   l_gxx * betaxz + l_gxy * betayz + l_gyz * betayx + l_gzz * betazx - l_gxz * betayy;
    Axx_rhs[idx] = chin1*src_xx + alpn1*(val_trK*l_Axx - TWO*term_xx) + TWO*(l_Axx*betaxx + l_Axy*betayx + l_Axz*betazx) - F2o3*l_Axx*div_beta;
    Ayy_rhs[idx] = chin1*src_yy + alpn1*(val_trK*l_Ayy - TWO*term_yy) + TWO*(l_Axy*betaxy + l_Ayy*betayy + l_Ayz*betazy) - F2o3*l_Ayy*div_beta;
    Azz_rhs[idx] = chin1*src_zz + alpn1*(val_trK*l_Azz - TWO*term_zz) + TWO*(l_Axz*betaxz + l_Ayz*betayz + l_Azz*betazz) - F2o3*l_Azz*div_beta;
    Axy_rhs[idx] = chin1*src_xy + alpn1*(val_trK*l_Axy - TWO*term_xy) + l_Axx*betaxy + l_Axz*betazy + l_Ayy*betayx + l_Ayz*betazx - l_Axy*betazz + F1o3*l_Axy*div_beta;
    Ayz_rhs[idx] = chin1*src_yz + alpn1*(val_trK*l_Ayz - TWO*term_yz) + l_Axy*betaxz + l_Ayy*betayz + l_Axz*betaxy + l_Azz*betazy - l_Ayz*betaxx + F1o3*l_Ayz*div_beta;
    Axz_rhs[idx] = chin1*src_xz + alpn1*(val_trK*l_Axz - TWO*term_xz) + l_Axx*betaxz + l_Axy*betayz + l_Ayz*betayx + l_Azz*betazx - l_Axz*betayy + F1o3*l_Axz*div_beta;
}

__global__ void rhs_source_gauge_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)eps; (void)co;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const double alpn1 = Lap[idx] + ONE;
    Lap_rhs[idx] = -TWO * alpn1 * trK[idx];
    betax_rhs[idx] = FF * dtSfx[idx];
    betay_rhs[idx] = FF * dtSfy[idx];
    betaz_rhs[idx] = FF * dtSfz[idx];
    dtSfx_rhs[idx] = Gamx_rhs[idx] - eta * dtSfx[idx];
    dtSfy_rhs[idx] = Gamy_rhs[idx] - eta * dtSfy[idx];
    dtSfz_rhs[idx] = Gamz_rhs[idx] - eta * dtSfz[idx];
}
// The advection and KO terms do not read the RHS argument inside
// The advection and KO terms do not read the RHS argument inside
// d_lopsided_point. Keeping this pass separate removes all derivative and
// curvature temporaries from its register footprint.
__global__ void rhs_advection_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)rho; (void)Sx; (void)Sy; (void)Sz;
    (void)Sxx; (void)Sxy; (void)Sxz; (void)Syy; (void)Syz; (void)Szz;
    (void)Gamxxx; (void)Gamxxy; (void)Gamxxz; (void)Gamxyy; (void)Gamxyz; (void)Gamxzz;
    (void)Gamyxx; (void)Gamyxy; (void)Gamyxz; (void)Gamyyy; (void)Gamyyz; (void)Gamyzz;
    (void)Gamzxx; (void)Gamzxy; (void)Gamzxz; (void)Gamzyy; (void)Gamzyz; (void)Gamzzz;
    (void)Rxx; (void)Rxy; (void)Rxz; (void)Ryy; (void)Ryz; (void)Rzz;
    (void)ham_Res; (void)movx_Res; (void)movy_Res; (void)movz_Res;
    (void)Gmx_Res; (void)Gmy_Res; (void)Gmz_Res; (void)co;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};
    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];
    const double d12dx = 1.0 / (12.0 * dX);
    const double d12dy = 1.0 / (12.0 * dY);
    const double d12dz = 1.0 / (12.0 * dZ);
    const int imax = ex0 - 1, jmax = ex1 - 1, kmax = ex2 - 1;
    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > 0 && fabs(Z[0]) < dZ) kmin = -3;
    if (symmetry > 1 && fabs(X[0]) < dX) imin = -3;
    if (symmetry > 1 && fabs(Y[0]) < dY) jmin = -3;
    const double vx = betax[idx], vy = betay[idx], vz = betaz[idx];

#define RHS_ADVECTION_LOPSIDED(field, s1, s2, s3) \
    d_lopsided_point(dims, field, vx, vy, vz, d12dx, d12dy, d12dz, \
                     imin, jmin, kmin, imax, jmax, kmax, symmetry, \
                     s1, s2, s3, i, j, k)

    // Metric variables.
    gxx_rhs[idx] += RHS_ADVECTION_LOPSIDED(dxx, SYM, SYM, SYM);
    if (eps > 0.0) gxx_rhs[idx] += d_kodis_point(dims, dxx, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);
    gxy_rhs[idx] += RHS_ADVECTION_LOPSIDED(gxy, ANTI, ANTI, SYM);
    if (eps > 0.0) gxy_rhs[idx] += d_kodis_point(dims, gxy, X, Y, Z, ANTI, ANTI, SYM, symmetry, eps, i, j, k);
    gxz_rhs[idx] += RHS_ADVECTION_LOPSIDED(gxz, ANTI, SYM, ANTI);
    if (eps > 0.0) gxz_rhs[idx] += d_kodis_point(dims, gxz, X, Y, Z, ANTI, SYM, ANTI, symmetry, eps, i, j, k);
    gyy_rhs[idx] += RHS_ADVECTION_LOPSIDED(dyy, SYM, SYM, SYM);
    if (eps > 0.0) gyy_rhs[idx] += d_kodis_point(dims, dyy, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);
    gyz_rhs[idx] += RHS_ADVECTION_LOPSIDED(gyz, SYM, ANTI, ANTI);
    if (eps > 0.0) gyz_rhs[idx] += d_kodis_point(dims, gyz, X, Y, Z, SYM, ANTI, ANTI, symmetry, eps, i, j, k);
    gzz_rhs[idx] += RHS_ADVECTION_LOPSIDED(dzz, SYM, SYM, SYM);
    if (eps > 0.0) gzz_rhs[idx] += d_kodis_point(dims, dzz, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // Extrinsic curvature.
    Axx_rhs[idx] += RHS_ADVECTION_LOPSIDED(Axx, SYM, SYM, SYM);
    if (eps > 0.0) Axx_rhs[idx] += d_kodis_point(dims, Axx, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);
    Axy_rhs[idx] += RHS_ADVECTION_LOPSIDED(Axy, ANTI, ANTI, SYM);
    if (eps > 0.0) Axy_rhs[idx] += d_kodis_point(dims, Axy, X, Y, Z, ANTI, ANTI, SYM, symmetry, eps, i, j, k);
    Axz_rhs[idx] += RHS_ADVECTION_LOPSIDED(Axz, ANTI, SYM, ANTI);
    if (eps > 0.0) Axz_rhs[idx] += d_kodis_point(dims, Axz, X, Y, Z, ANTI, SYM, ANTI, symmetry, eps, i, j, k);
    Ayy_rhs[idx] += RHS_ADVECTION_LOPSIDED(Ayy, SYM, SYM, SYM);
    if (eps > 0.0) Ayy_rhs[idx] += d_kodis_point(dims, Ayy, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);
    Ayz_rhs[idx] += RHS_ADVECTION_LOPSIDED(Ayz, SYM, ANTI, ANTI);
    if (eps > 0.0) Ayz_rhs[idx] += d_kodis_point(dims, Ayz, X, Y, Z, SYM, ANTI, ANTI, symmetry, eps, i, j, k);
    Azz_rhs[idx] += RHS_ADVECTION_LOPSIDED(Azz, SYM, SYM, SYM);
    if (eps > 0.0) Azz_rhs[idx] += d_kodis_point(dims, Azz, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);

    // Scalar and gauge variables.
    chi_rhs[idx] += RHS_ADVECTION_LOPSIDED(chi, SYM, SYM, SYM);
    if (eps > 0.0) chi_rhs[idx] += d_kodis_point(dims, chi, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);
    trK_rhs[idx] += RHS_ADVECTION_LOPSIDED(trK, SYM, SYM, SYM);
    if (eps > 0.0) trK_rhs[idx] += d_kodis_point(dims, trK, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);
    Gamx_rhs[idx] += RHS_ADVECTION_LOPSIDED(Gamx, ANTI, SYM, SYM);
    if (eps > 0.0) Gamx_rhs[idx] += d_kodis_point(dims, Gamx, X, Y, Z, ANTI, SYM, SYM, symmetry, eps, i, j, k);
    Gamy_rhs[idx] += RHS_ADVECTION_LOPSIDED(Gamy, SYM, ANTI, SYM);
    if (eps > 0.0) Gamy_rhs[idx] += d_kodis_point(dims, Gamy, X, Y, Z, SYM, ANTI, SYM, symmetry, eps, i, j, k);
    Gamz_rhs[idx] += RHS_ADVECTION_LOPSIDED(Gamz, SYM, SYM, ANTI);
    if (eps > 0.0) Gamz_rhs[idx] += d_kodis_point(dims, Gamz, X, Y, Z, SYM, SYM, ANTI, symmetry, eps, i, j, k);
    Lap_rhs[idx] += RHS_ADVECTION_LOPSIDED(Lap, SYM, SYM, SYM);
    if (eps > 0.0) Lap_rhs[idx] += d_kodis_point(dims, Lap, X, Y, Z, SYM, SYM, SYM, symmetry, eps, i, j, k);
    betax_rhs[idx] += RHS_ADVECTION_LOPSIDED(betax, ANTI, SYM, SYM);
    if (eps > 0.0) betax_rhs[idx] += d_kodis_point(dims, betax, X, Y, Z, ANTI, SYM, SYM, symmetry, eps, i, j, k);
    betay_rhs[idx] += RHS_ADVECTION_LOPSIDED(betay, SYM, ANTI, SYM);
    if (eps > 0.0) betay_rhs[idx] += d_kodis_point(dims, betay, X, Y, Z, SYM, ANTI, SYM, symmetry, eps, i, j, k);
    betaz_rhs[idx] += RHS_ADVECTION_LOPSIDED(betaz, SYM, SYM, ANTI);
    if (eps > 0.0) betaz_rhs[idx] += d_kodis_point(dims, betaz, X, Y, Z, SYM, SYM, ANTI, symmetry, eps, i, j, k);
    dtSfx_rhs[idx] += RHS_ADVECTION_LOPSIDED(dtSfx, ANTI, SYM, SYM);
    if (eps > 0.0) dtSfx_rhs[idx] += d_kodis_point(dims, dtSfx, X, Y, Z, ANTI, SYM, SYM, symmetry, eps, i, j, k);
    dtSfy_rhs[idx] += RHS_ADVECTION_LOPSIDED(dtSfy, SYM, ANTI, SYM);
    if (eps > 0.0) dtSfy_rhs[idx] += d_kodis_point(dims, dtSfy, X, Y, Z, SYM, ANTI, SYM, symmetry, eps, i, j, k);
    dtSfz_rhs[idx] += RHS_ADVECTION_LOPSIDED(dtSfz, SYM, SYM, ANTI);
    if (eps > 0.0) dtSfz_rhs[idx] += d_kodis_point(dims, dtSfz, X, Y, Z, SYM, SYM, ANTI, symmetry, eps, i, j, k);

#undef RHS_ADVECTION_LOPSIDED
}
// Constraints are evaluated only for the predictor stage. The kernel rebuilds
// the conformal connection needed by the Gamma residual, then consumes the
// physical connection/Ricci tensors produced by rhs_evolution_kernel.
__global__ void rhs_constraints_kernel(RHS_KERNEL_PARAMS) {
    (void)T; (void)Lap; (void)betax; (void)betay; (void)betaz;
    (void)dtSfx; (void)dtSfy; (void)dtSfz;
    (void)chi_rhs; (void)trK_rhs;
    (void)gxx_rhs; (void)gxy_rhs; (void)gxz_rhs; (void)gyy_rhs; (void)gyz_rhs; (void)gzz_rhs;
    (void)Axx_rhs; (void)Axy_rhs; (void)Axz_rhs; (void)Ayy_rhs; (void)Ayz_rhs; (void)Azz_rhs;
    (void)Gamx_rhs; (void)Gamy_rhs; (void)Gamz_rhs; (void)Lap_rhs;
    (void)betax_rhs; (void)betay_rhs; (void)betaz_rhs;
    (void)dtSfx_rhs; (void)dtSfy_rhs; (void)dtSfz_rhs;
    (void)Sxx; (void)Sxy; (void)Sxz; (void)Syy; (void)Syz; (void)Szz;
    if (co != 0 && co != RHS_CONSTRAINT_ONLY) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2};

    const double val_chi = chi[idx];
    const double chin1 = val_chi + ONE;
    const double val_trK = trK[idx];
    const double l_gxx = dxx[idx] + ONE;
    const double l_gxy = gxy[idx];
    const double l_gxz = gxz[idx];
    const double l_gyy = dyy[idx] + ONE;
    const double l_gyz = gyz[idx];
    const double l_gzz = dzz[idx] + ONE;
    const double l_Axx = Axx[idx];
    const double l_Axy = Axy[idx];
    const double l_Axz = Axz[idx];
    const double l_Ayy = Ayy[idx];
    const double l_Ayz = Ayz[idx];
    const double l_Azz = Azz[idx];

    double chix, chiy, chiz;
    d_fderivs_point(dims, chi, &chix, &chiy, &chiz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    // Metric derivatives and inverse metric for the Gamma constraint.
    double gxxx, gxxy, gxxz, gxyx, gxyy, gxyz, gxzx, gxzy, gxzz;
    double gyyx, gyyy, gyyz, gyzx, gyzy, gyzz, gzzx, gzzy, gzzz;
    d_fderivs_point(dims, dxx, &gxxx, &gxxy, &gxxz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gxy, &gxyx, &gxyy, &gxyz, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gxz, &gxzx, &gxzy, &gxzz, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, dyy, &gyyx, &gyyy, &gyyz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gyz, &gyzx, &gyzy, &gyzz, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, dzz, &gzzx, &gzzy, &gzzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    const double detg = l_gxx * (l_gyy * l_gzz - l_gyz * l_gyz)
                      - l_gxy * (l_gxy * l_gzz - l_gyz * l_gxz)
                      + l_gxz * (l_gxy * l_gyz - l_gyy * l_gxz);
    const double gupxx = (l_gyy * l_gzz - l_gyz * l_gyz) / detg;
    const double gupxy = -(l_gxy * l_gzz - l_gyz * l_gxz) / detg;
    const double gupxz = (l_gxy * l_gyz - l_gyy * l_gxz) / detg;
    const double gupyy = (l_gxx * l_gzz - l_gxz * l_gxz) / detg;
    const double gupyz = -(l_gxx * l_gyz - l_gxy * l_gxz) / detg;
    const double gupzz = (l_gxx * l_gyy - l_gxy * l_gxy) / detg;

    const double term_x = gupxx*(gupxx*gxxx+gupxy*gxyx+gupxz*gxzx)
                        + gupxy*(gupxx*gxyx+gupxy*gyyx+gupxz*gyzx)
                        + gupxz*(gupxx*gxzx+gupxy*gyzx+gupxz*gzzx)
                        + gupxx*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy)
                        + gupxy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy)
                        + gupxz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy)
                        + gupxx*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                        + gupxy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                        + gupxz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
    const double term_y = gupxx*(gupxy*gxxx+gupyy*gxyx+gupyz*gxzx)
                        + gupxy*(gupxy*gxyx+gupyy*gyyx+gupyz*gyzx)
                        + gupxz*(gupxy*gxzx+gupyy*gyzx+gupyz*gzzx)
                        + gupxy*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy)
                        + gupyy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy)
                        + gupyz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy)
                        + gupxy*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                        + gupyy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                        + gupyz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
    const double term_z = gupxx*(gupxz*gxxx+gupyz*gxyx+gupzz*gxzx)
                        + gupxy*(gupxz*gxyx+gupyz*gyyx+gupzz*gyzx)
                        + gupxz*(gupxz*gxzx+gupyz*gyzx+gupzz*gzzx)
                        + gupxy*(gupxz*gxxy+gupyz*gxyy+gupzz*gxzy)
                        + gupyy*(gupxz*gxyy+gupyz*gyyy+gupzz*gyzy)
                        + gupyz*(gupxz*gxzy+gupyz*gyzy+gupzz*gzzy)
                        + gupxz*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                        + gupyz*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                        + gupzz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
    Gmx_Res[idx] = Gamx[idx] - term_x;
    Gmy_Res[idx] = Gamy[idx] - term_y;
    Gmz_Res[idx] = Gamz[idx] - term_z;

    // Physical tensors are produced by the evolution pass on this stream.
    const double l_Gamxxx = Gamxxx[idx], l_Gamxxy = Gamxxy[idx], l_Gamxxz = Gamxxz[idx];
    const double l_Gamxyy = Gamxyy[idx], l_Gamxyz = Gamxyz[idx], l_Gamxzz = Gamxzz[idx];
    const double l_Gamyxx = Gamyxx[idx], l_Gamyxy = Gamyxy[idx], l_Gamyxz = Gamyxz[idx];
    const double l_Gamyyy = Gamyyy[idx], l_Gamyyz = Gamyyz[idx], l_Gamyzz = Gamyzz[idx];
    const double l_Gamzxx = Gamzxx[idx], l_Gamzxy = Gamzxy[idx], l_Gamzxz = Gamzxz[idx];
    const double l_Gamzyy = Gamzyy[idx], l_Gamzyz = Gamzyz[idx], l_Gamzzz = Gamzzz[idx];
    const double l_Rxx = Rxx[idx], l_Rxy = Rxy[idx], l_Rxz = Rxz[idx];
    const double l_Ryy = Ryy[idx], l_Ryz = Ryz[idx], l_Rzz = Rzz[idx];

    const double ham_val = gupxx*l_Rxx + gupyy*l_Ryy + gupzz*l_Rzz
                         + TWO*(gupxy*l_Rxy + gupxz*l_Rxz + gupyz*l_Ryz);
    const double termA_xx = gupxx*l_Axx*l_Axx + gupyy*l_Axy*l_Axy + gupzz*l_Axz*l_Axz
                          + TWO*(gupxy*l_Axx*l_Axy + gupxz*l_Axx*l_Axz + gupyz*l_Axy*l_Axz);
    const double termA_yy = gupxx*l_Axy*l_Axy + gupyy*l_Ayy*l_Ayy + gupzz*l_Ayz*l_Ayz
                          + TWO*(gupxy*l_Axy*l_Ayy + gupxz*l_Axy*l_Ayz + gupyz*l_Ayy*l_Ayz);
    const double termA_zz = gupxx*l_Axz*l_Axz + gupyy*l_Ayz*l_Ayz + gupzz*l_Azz*l_Azz
                          + TWO*(gupxy*l_Axz*l_Ayz + gupxz*l_Axz*l_Azz + gupyz*l_Ayz*l_Azz);
    const double termA_xy = gupxx*l_Axx*l_Axy + gupyy*l_Axy*l_Ayy + gupzz*l_Axz*l_Ayz
                          + gupxy*(l_Axx*l_Ayy + l_Axy*l_Axy)
                          + gupxz*(l_Axx*l_Ayz + l_Axz*l_Axy)
                          + gupyz*(l_Axy*l_Ayz + l_Axz*l_Ayy);
    const double termA_xz = gupxx*l_Axx*l_Axz + gupyy*l_Axy*l_Ayz + gupzz*l_Axz*l_Azz
                          + gupxy*(l_Axx*l_Ayz + l_Axy*l_Axz)
                          + gupxz*(l_Axx*l_Azz + l_Axz*l_Axz)
                          + gupyz*(l_Axy*l_Azz + l_Axz*l_Ayz);
    const double termA_yz = gupxx*l_Axy*l_Axz + gupyy*l_Ayy*l_Ayz + gupzz*l_Ayz*l_Azz
                          + gupxy*(l_Axy*l_Ayz + l_Ayy*l_Axz)
                          + gupxz*(l_Axy*l_Azz + l_Ayz*l_Axz)
                          + gupyz*(l_Ayy*l_Azz + l_Ayz*l_Ayz);
    const double trA2 = gupxx*termA_xx + gupyy*termA_yy + gupzz*termA_zz
                      + TWO*(gupxy*termA_xy + gupxz*termA_xz + gupyz*termA_yz);
    ham_Res[idx] = chin1*ham_val + F2o3*val_trK*val_trK - trA2 - F16*PI*rho[idx];

    double Kx, Ky, Kz;
    d_fderivs_point(dims, trK, &Kx, &Ky, &Kz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    double d_Axx_x, d_Axx_y, d_Axx_z, d_Axy_x, d_Axy_y, d_Axy_z;
    double d_Axz_x, d_Axz_y, d_Axz_z, d_Ayy_x, d_Ayy_y, d_Ayy_z;
    double d_Ayz_x, d_Ayz_y, d_Ayz_z, d_Azz_x, d_Azz_y, d_Azz_z;
    d_fderivs_point(dims, Axx, &d_Axx_x, &d_Axx_y, &d_Axx_z, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Axy, &d_Axy_x, &d_Axy_y, &d_Axy_z, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Axz, &d_Axz_x, &d_Axz_y, &d_Axz_z, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Ayy, &d_Ayy_x, &d_Ayy_y, &d_Ayy_z, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Ayz, &d_Ayz_x, &d_Ayz_y, &d_Ayz_z, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Azz, &d_Azz_x, &d_Azz_y, &d_Azz_z, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    const double DA_xxx = d_Axx_x - TWO*(l_Gamxxx*l_Axx + l_Gamyxx*l_Axy + l_Gamzxx*l_Axz) - chix*l_Axx/chin1;
    const double DA_xyx = d_Axy_x - (l_Gamxxy*l_Axx + l_Gamyxy*l_Axy + l_Gamzxy*l_Axz + l_Gamxxx*l_Axy + l_Gamyxx*l_Ayy + l_Gamzxx*l_Ayz) - chix*l_Axy/chin1;
    const double DA_xzx = d_Axz_x - (l_Gamxxz*l_Axx + l_Gamyxz*l_Axy + l_Gamzxz*l_Axz + l_Gamxxx*l_Axz + l_Gamyxx*l_Ayz + l_Gamzxx*l_Azz) - chix*l_Axz/chin1;
    const double DA_yyx = d_Ayy_x - TWO*(l_Gamxxy*l_Axy + l_Gamyxy*l_Ayy + l_Gamzxy*l_Ayz) - chix*l_Ayy/chin1;
    const double DA_yzx = d_Ayz_x - (l_Gamxxz*l_Axy + l_Gamyxz*l_Ayy + l_Gamzxz*l_Ayz + l_Gamxxy*l_Axz + l_Gamyxy*l_Ayz + l_Gamzxy*l_Azz) - chix*l_Ayz/chin1;
    const double DA_zzx = d_Azz_x - TWO*(l_Gamxxz*l_Axz + l_Gamyxz*l_Ayz + l_Gamzxz*l_Azz) - chix*l_Azz/chin1;
    const double DA_xxy = d_Axx_y - TWO*(l_Gamxxy*l_Axx + l_Gamyxy*l_Axy + l_Gamzxy*l_Axz) - chiy*l_Axx/chin1;
    const double DA_xyy = d_Axy_y - (l_Gamxyy*l_Axx + l_Gamyyy*l_Axy + l_Gamzyy*l_Axz + l_Gamxxy*l_Axy + l_Gamyxy*l_Ayy + l_Gamzxy*l_Ayz) - chiy*l_Axy/chin1;
    const double DA_xzy = d_Axz_y - (l_Gamxyz*l_Axx + l_Gamyyz*l_Axy + l_Gamzyz*l_Axz + l_Gamxxy*l_Axz + l_Gamyxy*l_Ayz + l_Gamzxy*l_Azz) - chiy*l_Axz/chin1;
    const double DA_yyy = d_Ayy_y - TWO*(l_Gamxyy*l_Axy + l_Gamyyy*l_Ayy + l_Gamzyy*l_Ayz) - chiy*l_Ayy/chin1;
    const double DA_yzy = d_Ayz_y - (l_Gamxyz*l_Axy + l_Gamyyz*l_Ayy + l_Gamzyz*l_Ayz + l_Gamxyy*l_Axz + l_Gamyyy*l_Ayz + l_Gamzyy*l_Azz) - chiy*l_Ayz/chin1;
    const double DA_zzy = d_Azz_y - TWO*(l_Gamxyz*l_Axz + l_Gamyyz*l_Ayz + l_Gamzyz*l_Azz) - chiy*l_Azz/chin1;
    const double DA_xxz = d_Axx_z - TWO*(l_Gamxxz*l_Axx + l_Gamyxz*l_Axy + l_Gamzxz*l_Axz) - chiz*l_Axx/chin1;
    const double DA_xyz = d_Axy_z - (l_Gamxyz*l_Axx + l_Gamyyz*l_Axy + l_Gamzyz*l_Axz + l_Gamxxz*l_Axy + l_Gamyxz*l_Ayy + l_Gamzxz*l_Ayz) - chiz*l_Axy/chin1;
    const double DA_xzz = d_Axz_z - (l_Gamxzz*l_Axx + l_Gamyzz*l_Axy + l_Gamzzz*l_Axz + l_Gamxxz*l_Axz + l_Gamyxz*l_Ayz + l_Gamzxz*l_Azz) - chiz*l_Axz/chin1;
    const double DA_yyz = d_Ayy_z - TWO*(l_Gamxyz*l_Axy + l_Gamyyz*l_Ayy + l_Gamzyz*l_Ayz) - chiz*l_Ayy/chin1;
    const double DA_yzz = d_Ayz_z - (l_Gamxzz*l_Axy + l_Gamyzz*l_Ayy + l_Gamzzz*l_Ayz + l_Gamxyz*l_Axz + l_Gamyyz*l_Ayz + l_Gamzyz*l_Azz) - chiz*l_Ayz/chin1;
    const double DA_zzz = d_Azz_z - TWO*(l_Gamxzz*l_Axz + l_Gamyzz*l_Ayz + l_Gamzzz*l_Azz) - chiz*l_Azz/chin1;

    movx_Res[idx] = gupxx*DA_xxx + gupyy*DA_xyy + gupzz*DA_xzz
                  + gupxy*DA_xyx + gupxz*DA_xzx + gupyz*DA_xzy
                  + gupxy*DA_xxy + gupxz*DA_xxz + gupyz*DA_xyz
                  - F2o3*Kx - F8*PI*Sx[idx];
    movy_Res[idx] = gupxx*DA_xyx + gupyy*DA_yyy + gupzz*DA_yzz
                  + gupxy*DA_yyx + gupxz*DA_yzx + gupyz*DA_yzy
                  + gupxy*DA_xyy + gupxz*DA_xyz + gupyz*DA_yyz
                  - F2o3*Ky - F8*PI*Sy[idx];
    movz_Res[idx] = gupxx*DA_xzx + gupyy*DA_yzy + gupzz*DA_zzz
                  + gupxy*DA_yzx + gupxz*DA_zzx + gupyz*DA_zzy
                  + gupxy*DA_xzy + gupxz*DA_xzz + gupyz*DA_yzz
                  - F2o3*Kz - F8*PI*Sz[idx];
}
#define RHS_LAUNCH_ARGS \
        ex[0], ex[1], ex[2], T, d_X, d_Y, d_Z, \
        d_chi, d_trK, \
        d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz, \
        d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz, \
        d_Gamx, d_Gamy, d_Gamz, d_Lap, \
        d_betax, d_betay, d_betaz, d_dtSfx, d_dtSfy, d_dtSfz, \
        d_chi_rhs, d_trK_rhs, \
        d_gxx_rhs, d_gxy_rhs, d_gxz_rhs, d_gyy_rhs, d_gyz_rhs, d_gzz_rhs, \
        d_Axx_rhs, d_Axy_rhs, d_Axz_rhs, d_Ayy_rhs, d_Ayz_rhs, d_Azz_rhs, \
        d_Gamx_rhs, d_Gamy_rhs, d_Gamz_rhs, d_Lap_rhs, \
        d_betax_rhs, d_betay_rhs, d_betaz_rhs, d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs, \
        d_rho, d_Sx, d_Sy, d_Sz, d_Sxx, d_Sxy, d_Sxz, d_Syy, d_Syz, d_Szz, \
        d_Gamxxx, d_Gamxxy, d_Gamxxz, d_Gamxyy, d_Gamxyz, d_Gamxzz, \
        d_Gamyxx, d_Gamyxy, d_Gamyxz, d_Gamyyy, d_Gamyyz, d_Gamyzz, \
        d_Gamzxx, d_Gamzxy, d_Gamzxz, d_Gamzyy, d_Gamzyz, d_Gamzzz, \
        d_Rxx, d_Rxy, d_Rxz, d_Ryy, d_Ryz, d_Rzz, \
        d_ham_Res, d_movx_Res, d_movy_Res, d_movz_Res, \
        d_Gmx_Res, d_Gmy_Res, d_Gmz_Res, symmetry, lev, eps, co
void gpu_compute_rhs_bssn_launch( // launch kernel with device pointers
    cudaStream_t &stream,
    int* ex, double T, double* d_X, double* d_Y, double* d_Z,
    double* d_chi, double* d_trK,
    double* d_dxx, double* d_gxy, double* d_gxz,
    double* d_dyy, double* d_gyz, double* d_dzz,
    double* d_Axx, double* d_Axy, double* d_Axz,
    double* d_Ayy, double* d_Ayz, double* d_Azz,
    double* d_Gamx, double* d_Gamy, double* d_Gamz,
    double* d_Lap,
    double* d_betax, double* d_betay, double* d_betaz,
    double* d_dtSfx, double* d_dtSfy, double* d_dtSfz,
    double* d_chi_rhs, double* d_trK_rhs,
    double* d_gxx_rhs, double* d_gxy_rhs, double* d_gxz_rhs,
    double* d_gyy_rhs, double* d_gyz_rhs, double* d_gzz_rhs,
    double* d_Axx_rhs, double* d_Axy_rhs, double* d_Axz_rhs,
    double* d_Ayy_rhs, double* d_Ayz_rhs, double* d_Azz_rhs,
    double* d_Gamx_rhs, double* d_Gamy_rhs, double* d_Gamz_rhs,
    double* d_Lap_rhs,
    double* d_betax_rhs, double* d_betay_rhs, double* d_betaz_rhs,
    double* d_dtSfx_rhs, double* d_dtSfy_rhs, double* d_dtSfz_rhs,
    double* d_rho, double* d_Sx, double* d_Sy, double* d_Sz,
    double* d_Sxx, double* d_Sxy, double* d_Sxz,
    double* d_Syy, double* d_Syz, double* d_Szz,
    double* d_Gamxxx, double* d_Gamxxy, double* d_Gamxxz,
    double* d_Gamxyy, double* d_Gamxyz, double* d_Gamxzz,
    double* d_Gamyxx, double* d_Gamyxy, double* d_Gamyxz,
    double* d_Gamyyy, double* d_Gamyyz, double* d_Gamyzz,
    double* d_Gamzxx, double* d_Gamzxy, double* d_Gamzxz,
    double* d_Gamzyy, double* d_Gamzyz, double* d_Gamzzz,
    double* d_Rxx, double* d_Rxy, double* d_Rxz,
    double* d_Ryy, double* d_Ryz, double* d_Rzz,
    double* d_ham_Res, double* d_movx_Res, double* d_movy_Res, double* d_movz_Res,
    double* d_Gmx_Res, double* d_Gmy_Res, double* d_Gmz_Res,
    int symmetry, int lev, double eps, int co
) {
    const bool constraint_only = (co == RHS_CONSTRAINT_ONLY);
    dim3 block(8, 8, 4); // 调整 block size 以适应架构
    dim3 grid(
        (ex[0] + block.x - 1) / block.x,
        (ex[1] + block.y - 1) / block.y,
        (ex[2] + block.z - 1) / block.z
    );

    // Geometry producer must complete before the evolution consumer reads its scratch.
    rhs_geometry_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
    rhs_ricci_a_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
    rhs_gamma_seed_fused_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
    // 1. Kernel 1: Derivatives & Connection Coefficients
    if (symmetry == 1) {
        CompactBetaPrepareFields beta_prepare_fields{};
        const double* beta_inputs[COMPACT_BETA_FIELDS] = {
            d_betax, d_betay, d_betaz
        };
        double* beta_divergence[COMPACT_BETA_FIELDS] = {
            d_ham_Res, d_movx_Res, d_movy_Res
        };
        double* beta_laplacian[COMPACT_BETA_FIELDS] = {
            d_movz_Res, d_Gmx_Res, d_Gmy_Res
        };
        const int beta_x_parity[COMPACT_BETA_FIELDS] = {-1, 1, 1};
        const int beta_y_parity[COMPACT_BETA_FIELDS] = {1, -1, 1};
        const int beta_z_parity[COMPACT_BETA_FIELDS] = {1, 1, -1};
        for (int field = 0; field < COMPACT_BETA_FIELDS; ++field) {
            beta_prepare_fields.input[field] = beta_inputs[field];
            beta_prepare_fields.divergence[field] = beta_divergence[field];
            beta_prepare_fields.laplacian[field] = beta_laplacian[field];
            beta_prepare_fields.parity_x[field] = beta_x_parity[field];
            beta_prepare_fields.parity_y[field] = beta_y_parity[field];
            beta_prepare_fields.parity_z[field] = beta_z_parity[field];
        }

        CompactConnectionFields connection_fields{};
        const double* connection_inputs[COMPACT_BETA_FIELDS]
            [COMPACT_CONNECTION_COMPONENTS] = {
                {d_Gamxxx, d_Gamxxy, d_Gamxxz, d_Gamxyy, d_Gamxyz, d_Gamxzz},
                {d_Gamyxx, d_Gamyxy, d_Gamyxz, d_Gamyyy, d_Gamyyz, d_Gamyzz},
                {d_Gamzxx, d_Gamzxy, d_Gamzxz, d_Gamzyy, d_Gamzyz, d_Gamzzz}
            };
        double* connection_outputs[COMPACT_BETA_FIELDS] = {
            d_Ayy_rhs, d_Ayz_rhs, d_Azz_rhs
        };
        for (int upper = 0; upper < COMPACT_BETA_FIELDS; ++upper) {
            for (int component = 0;
                 component < COMPACT_CONNECTION_COMPONENTS; ++component) {
                connection_fields.input[upper][component] =
                    connection_inputs[upper][component];
            }
            connection_fields.contracted[upper] = connection_outputs[upper];
        }

        launch_rhs_beta_gamma_prepare_equatorial_compact(
            stream, ex[0], ex[1], ex[2], d_X, d_Y, d_Z,
            d_betax_rhs, d_betay_rhs, d_betaz_rhs,
            d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs,
            beta_prepare_fields, connection_fields
        );
    } else {
        rhs_beta_gamma_prepare_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
    }
    rhs_beta_gamma_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
    if (symmetry == 1) {
        CompactHessianFields hessian_fields{};
        const double* hessian_inputs[COMPACT_HESSIAN_FIELDS] = {
            d_dxx, d_dyy, d_dzz, d_gxy, d_gxz, d_gyz
        };
        double* hessian_outputs[COMPACT_HESSIAN_FIELDS] = {
            d_Rxx, d_Ryy, d_Rzz, d_Rxy, d_Rxz, d_Ryz
        };
        const int hessian_x_parity[COMPACT_HESSIAN_FIELDS] = {
            1, 1, 1, -1, -1, 1
        };
        const int hessian_y_parity[COMPACT_HESSIAN_FIELDS] = {
            1, 1, 1, -1, 1, -1
        };
        const int hessian_z_parity[COMPACT_HESSIAN_FIELDS] = {
            1, 1, 1, 1, -1, -1
        };
        for (int field = 0; field < COMPACT_HESSIAN_FIELDS; ++field) {
            hessian_fields.input[field] = hessian_inputs[field];
            hessian_fields.output[field] = hessian_outputs[field];
            hessian_fields.parity_x[field] = hessian_x_parity[field];
            hessian_fields.parity_y[field] = hessian_y_parity[field];
            hessian_fields.parity_z[field] = hessian_z_parity[field];
        }
        launch_rhs_evolution_equatorial_compact(
            stream, ex[0], ex[1], ex[2], d_X, d_Y, d_Z,
            d_betax_rhs, d_betay_rhs, d_betaz_rhs,
            d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs,
            hessian_fields
        );
    } else {
        rhs_evolution_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
    }
    rhs_ricci_connection_diag_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
    rhs_ricci_connection_offdiag_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
    // Source and advection mutate RHS arrays but are not consumed by constraints.
    if (!constraint_only) {
        // Keep the geometry-stage Chi gradient alive until all Chi source
        // consumers finish.  Other symmetry modes retain the legacy order.
        if (symmetry == 1) {
            const double* source_connections[3][COMPACT_TENSOR_COMPONENTS] = {
                {d_Gamxxx, d_Gamxxy, d_Gamxxz, d_Gamxyy, d_Gamxyz, d_Gamxzz},
                {d_Gamyxx, d_Gamyxy, d_Gamyxz, d_Gamyyy, d_Gamyyz, d_Gamyzz},
                {d_Gamzxx, d_Gamzxy, d_Gamzxz, d_Gamzyy, d_Gamzyz, d_Gamzzz}
            };
            double* source_hessian[COMPACT_TENSOR_COMPONENTS] = {
                d_ham_Res, d_movx_Res, d_movy_Res,
                d_movz_Res, d_Gmx_Res, d_Gmy_Res
            };

            CompactChiHessianFields chi_fields{};
            chi_fields.gradient[0] = d_chi_rhs;
            chi_fields.gradient[1] = d_trK_rhs;
            chi_fields.gradient[2] = d_Lap_rhs;
            for (int upper = 0; upper < 3; ++upper) {
                for (int component = 0;
                     component < COMPACT_TENSOR_COMPONENTS; ++component) {
                    chi_fields.connection[upper][component] =
                        source_connections[upper][component];
                }
            }
            for (int component = 0;
                 component < COMPACT_TENSOR_COMPONENTS; ++component) {
                chi_fields.covariant_hessian[component] =
                    source_hessian[component];
            }
            launch_rhs_source_chi_hessian_equatorial_compact(
                stream, ex[0], ex[1], ex[2], d_X, d_Y, d_Z,
                d_chi, chi_fields
            );
            rhs_source_chi_ricci_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
            rhs_source_physical_gamma_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);

            CompactLapseHessianFields lapse_fields{};
            const double* inverse_metric[COMPACT_TENSOR_COMPONENTS] = {
                d_betax_rhs, d_betay_rhs, d_betaz_rhs,
                d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs
            };
            for (int upper = 0; upper < 3; ++upper) {
                for (int component = 0;
                     component < COMPACT_TENSOR_COMPONENTS; ++component) {
                    lapse_fields.connection[upper][component] =
                        source_connections[upper][component];
                }
            }
            for (int component = 0;
                 component < COMPACT_TENSOR_COMPONENTS; ++component) {
                lapse_fields.inverse_metric[component] = inverse_metric[component];
                lapse_fields.covariant_hessian[component] = source_hessian[component];
            }
            lapse_fields.trace = d_Gmz_Res;
            launch_rhs_source_lapse_equatorial_compact(
                stream, ex[0], ex[1], ex[2], d_X, d_Y, d_Z,
                d_Lap, lapse_fields
            );
        } else {
            rhs_source_metric_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
            rhs_source_chi_hessian_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
            rhs_source_chi_ricci_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
            rhs_source_physical_gamma_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
            rhs_source_lapse_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
        }
        // Gauge runs last because the Aij kernels still consume inverse metric values in the gauge RHS slots.
        rhs_source_trace_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
        if (symmetry == 1) {
            rhs_source_metric_a_equatorial_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
        } else {
            rhs_source_a_diag_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
            rhs_source_a_offdiag_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
        }
        rhs_source_gauge_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);

        // Kernel 2: use the equatorial tiled path only for the course's
        // equatorial-symmetry mode; retain the legacy path for other modes.
        if (symmetry == 1) {
            CompactAdvectionFields compact_fields{};
            const double* compact_inputs[COMPACT_ADVECTION_FIELDS] = {
                d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
                d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
                d_chi, d_trK, d_Gamx, d_Gamy, d_Gamz, d_Lap,
                d_betax, d_betay, d_betaz, d_dtSfx, d_dtSfy, d_dtSfz
            };
            double* compact_rhs[COMPACT_ADVECTION_FIELDS] = {
                d_gxx_rhs, d_gxy_rhs, d_gxz_rhs, d_gyy_rhs, d_gyz_rhs, d_gzz_rhs,
                d_Axx_rhs, d_Axy_rhs, d_Axz_rhs, d_Ayy_rhs, d_Ayz_rhs, d_Azz_rhs,
                d_chi_rhs, d_trK_rhs, d_Gamx_rhs, d_Gamy_rhs, d_Gamz_rhs, d_Lap_rhs,
                d_betax_rhs, d_betay_rhs, d_betaz_rhs,
                d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs
            };
            const int compact_x_parity[COMPACT_ADVECTION_FIELDS] = {
                1, -1, -1, 1, 1, 1,
                1, -1, -1, 1, 1, 1,
                1, 1, -1, 1, 1, 1,
                -1, 1, 1, -1, 1, 1
            };
            const int compact_y_parity[COMPACT_ADVECTION_FIELDS] = {
                1, -1, 1, 1, -1, 1,
                1, -1, 1, 1, -1, 1,
                1, 1, 1, -1, 1, 1,
                1, -1, 1, 1, -1, 1
            };
            const int compact_z_parity[COMPACT_ADVECTION_FIELDS] = {
                1, 1, -1, 1, -1, 1,
                1, 1, -1, 1, -1, 1,
                1, 1, 1, 1, -1, 1,
                1, 1, -1, 1, 1, -1
            };
            for (int field = 0; field < COMPACT_ADVECTION_FIELDS; ++field) {
                compact_fields.input[field] = compact_inputs[field];
                compact_fields.rhs[field] = compact_rhs[field];
                compact_fields.parity_x[field] = compact_x_parity[field];
                compact_fields.parity_y[field] = compact_y_parity[field];
                compact_fields.parity_z[field] = compact_z_parity[field];
            }
            launch_rhs_advection_equatorial_compact(
                stream, ex[0], ex[1], ex[2],
                d_X, d_Y, d_Z, d_betax, d_betay, d_betaz,
                compact_fields, eps
            );
        } else {
            rhs_advection_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
        }
    }

    // Kernel 3: constraints are only needed for the predictor stage.
    if (co == 0 || constraint_only) {
        rhs_constraints_kernel<<<grid, block, 0, stream>>>(RHS_LAUNCH_ARGS);
    }
#undef RHS_LAUNCH_ARGS
}
