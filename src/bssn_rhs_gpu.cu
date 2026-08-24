#include "bssn_rhs.h"

#include "fmisc.h"
#include "derivatives.h"
#include "kodiss.h"
#include "lopsidediff.h"
#include "gpu_manager.h"

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

__global__ void rhs_kernel(
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

    // ==========================================
    // 1. 读取基础变量并进行代数变换
    // ==========================================
    // Fortran: alpn1 = Lap + ONE, chin1 = chi + ONE
    double val_Lap = Lap[idx];
    double val_chi = chi[idx];
    double alpn1 = val_Lap + ONE;
    double chin1 = val_chi + ONE;

    // Metric (dxx 是偏差量，gxx 是物理量 gxx = dxx + 1)
    double val_gxx = dxx[idx] + ONE;
    double val_gxy = gxy[idx];
    double val_gxz = gxz[idx];
    double val_gyy = dyy[idx] + ONE;
    double val_gyz = gyz[idx];
    double val_gzz = dzz[idx] + ONE;

    double val_trK = trK[idx];

    // ==========================================
    // 3. 计算 Chi 的导数与 RHS
    // ==========================================
    double chix, chiy, chiz;
    d_fderivs_point(dims, chi, &chix, &chiy, &chiz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    // ==========================================
    // 4. 计算 Metric (gij) 导数
    // ==========================================
    double gxxx, gxxy, gxxz;
    double gxyx, gxyy, gxyz;
    double gxzx, gxzy, gxzz;
    double gyyx, gyyy, gyyz;
    double gyzx, gyzy, gyzz;
    double gzzx, gzzy, gzzz;

    d_fderivs_point(dims, dxx, &gxxx, &gxxy, &gxxz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gxy, &gxyx, &gxyy, &gxyz, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gxz, &gxzx, &gxzy, &gxzz, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, dyy, &gyyx, &gyyy, &gyyz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, gyz, &gyzx, &gyzy, &gyzz, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    d_fderivs_point(dims, dzz, &gzzx, &gzzy, &gzzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    // gxxx_out[idx] = gxxx; gxxy_out[idx] = gxxy; gxxz_out[idx] = gxxz;
    // gxyx_out[idx] = gxyx; gxyy_out[idx] = gxyy; gxyz_out[idx] = gxyz;
    // gxzx_out[idx] = gxzx; gxzy_out[idx] = gxzy; gxzz_out[idx] = gxzz;
    // gyyx_out[idx] = gyyx; gyyy_out[idx] = gyyy; gyyz_out[idx] = gyyz;
    // gyzx_out[idx] = gyzx; gyzy_out[idx] = gyzy; gyzz_out[idx] = gyzz;
    // gzzx_out[idx] = gzzx; gzzy_out[idx] = gzzy; gzzz_out[idx] = gzzz;

    // ==========================================
    // 6. 计算逆度规 (Invert Tilted Metric)
    // ==========================================
    double gupzz = val_gxx * val_gyy * val_gzz + val_gxy * val_gyz * val_gxz + val_gxz * val_gxy * val_gyz -
                   val_gxz * val_gyy * val_gxz - val_gxy * val_gxy * val_gzz - val_gxx * val_gyz * val_gyz;
    
    double gupxx = (val_gyy * val_gzz - val_gyz * val_gyz) / gupzz;
    double gupxy = -(val_gxy * val_gzz - val_gyz * val_gxz) / gupzz;
    double gupxz = (val_gxy * val_gyz - val_gyy * val_gxz) / gupzz;
    double gupyy = (val_gxx * val_gzz - val_gxz * val_gxz) / gupzz;
    double gupyz = -(val_gxx * val_gyz - val_gxy * val_gxz) / gupzz;
    gupzz = (val_gxx * val_gyy - val_gxy * val_gxy) / gupzz; // 更新 gupzz 为逆分量

    // 写回 gup
    // gupxx_out[idx] = gupxx; gupxy_out[idx] = gupxy; gupxz_out[idx] = gupxz;
    // gupyy_out[idx] = gupyy; gupyz_out[idx] = gupyz; gupzz_out[idx] = gupzz;

    // ==========================================
    // 7. 计算连接系数残差 (仅 co == 0)
    // ==========================================
    if (co == 0) {
        // Gmx_Res
        double term_x = gupxx*(gupxx*gxxx+gupxy*gxyx+gupxz*gxzx)
                      + gupxy*(gupxx*gxyx+gupxy*gyyx+gupxz*gyzx)
                      + gupxz*(gupxx*gxzx+gupxy*gyzx+gupxz*gzzx)
                      + gupxx*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy)
                      + gupxy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy)
                      + gupxz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy)
                      + gupxx*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                      + gupxy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                      + gupxz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
        Gmx_Res[idx] = Gamx[idx] - term_x;

        // Gmy_Res
        double term_y = gupxx*(gupxy*gxxx+gupyy*gxyx+gupyz*gxzx)
                      + gupxy*(gupxy*gxyx+gupyy*gyyx+gupyz*gyzx)
                      + gupxz*(gupxy*gxzx+gupyy*gyzx+gupyz*gzzx)
                      + gupxy*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy)
                      + gupyy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy)
                      + gupyz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy)
                      + gupxy*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                      + gupyy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                      + gupyz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
        Gmy_Res[idx] = Gamy[idx] - term_y;

        // Gmz_Res
        double term_z = gupxx*(gupxz*gxxx+gupyz*gxyx+gupzz*gxzx)
                      + gupxy*(gupxz*gxyx+gupyz*gyyx+gupzz*gyzx)
                      + gupxz*(gupxz*gxzx+gupyz*gyzx+gupzz*gzzx)
                      + gupxy*(gupxz*gxxy+gupyz*gxyy+gupzz*gxzy)
                      + gupyy*(gupxz*gxyy+gupyz*gyyy+gupzz*gyzy)
                      + gupyz*(gupxz*gxzy+gupyz*gyzy+gupzz*gzzy)
                      + gupxz*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                      + gupyz*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                      + gupzz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
        Gmz_Res[idx] = Gamz[idx] - term_z;
    }

    // ==========================================
    // 8. 计算第二类 Christoffel 符号 (Gam^k_ij)
    // ==========================================
    double l_Gamxxx; double l_Gamxxy; double l_Gamxxz;
    double l_Gamxyy; double l_Gamxyz; double l_Gamxzz;
    double l_Gamyxx; double l_Gamyxy; double l_Gamyxz;
    double l_Gamyyy; double l_Gamyyz; double l_Gamyzz;
    double l_Gamzxx; double l_Gamzxy; double l_Gamzxz;
    double l_Gamzyy; double l_Gamzyz; double l_Gamzzz;

    l_Gamxxx = HALF * (gupxx * gxxx + gupxy * (TWO * gxyx - gxxy) + gupxz * (TWO * gxzx - gxxz));
    l_Gamyxx = HALF * (gupxy * gxxx + gupyy * (TWO * gxyx - gxxy) + gupyz * (TWO * gxzx - gxxz));
    l_Gamzxx = HALF * (gupxz * gxxx + gupyz * (TWO * gxyx - gxxy) + gupzz * (TWO * gxzx - gxxz));

    l_Gamxyy = HALF * (gupxx * (TWO * gxyy - gyyx) + gupxy * gyyy + gupxz * (TWO * gyzy - gyyz));
    l_Gamyyy = HALF * (gupxy * (TWO * gxyy - gyyx) + gupyy * gyyy + gupyz * (TWO * gyzy - gyyz));
    l_Gamzyy = HALF * (gupxz * (TWO * gxyy - gyyx) + gupyz * gyyy + gupzz * (TWO * gyzy - gyyz));

    l_Gamxzz = HALF * (gupxx * (TWO * gxzz - gzzx) + gupxy * (TWO * gyzz - gzzy) + gupxz * gzzz);
    l_Gamyzz = HALF * (gupxy * (TWO * gxzz - gzzx) + gupyy * (TWO * gyzz - gzzy) + gupyz * gzzz);
    l_Gamzzz = HALF * (gupxz * (TWO * gxzz - gzzx) + gupyz * (TWO * gyzz - gzzy) + gupzz * gzzz);

    l_Gamxxy = HALF * (gupxx * gxxy + gupxy * gyyx + gupxz * (gxzy + gyzx - gxyz));
    l_Gamyxy = HALF * (gupxy * gxxy + gupyy * gyyx + gupyz * (gxzy + gyzx - gxyz));
    l_Gamzxy = HALF * (gupxz * gxxy + gupyz * gyyx + gupzz * (gxzy + gyzx - gxyz));

    l_Gamxxz = HALF * (gupxx * gxxz + gupxy * (gxyz + gyzx - gxzy) + gupxz * gzzx);
    l_Gamyxz = HALF * (gupxy * gxxz + gupyy * (gxyz + gyzx - gxzy) + gupyz * gzzx);
    l_Gamzxz = HALF * (gupxz * gxxz + gupyz * (gxyz + gyzx - gxzy) + gupzz * gzzx);

    l_Gamxyz = HALF * (gupxx * (gxyz + gxzy - gyzx) + gupxy * gyyz + gupxz * gzzy);
    l_Gamyyz = HALF * (gupxy * (gxyz + gxzy - gyzx) + gupyy * gyyz + gupyz * gzzy);
    l_Gamzyz = HALF * (gupxz * (gxyz + gxzy - gyzx) + gupyz * gyyz + gupzz * gzzy);

    // ------------------------------------------------------------------------------------
    // bssn_rhs_core_kernel
    // ------------------------------------------------------------------------------------

    // ==========================================
    // 0. 加载数据至寄存器 (Locals)
    // ==========================================
    double l_gxx = dxx[idx] + ONE; double l_gxy = gxy[idx]; double l_gxz = gxz[idx];
    double l_gyy = dyy[idx] + ONE; double l_gyz = gyz[idx]; double l_gzz = dzz[idx] + ONE;

    // double gupxx = gupxx_in[idx]; double gupxy = gupxy_in[idx]; double gupxz = gupxz_in[idx];
    // double gupyy = gupyy_in[idx]; double gupyz = gupyz_in[idx]; double gupzz = gupzz_in[idx];

    // double l_Gamxxx = l_Gamxxx; double l_Gamxxy = l_Gamxxy; double l_Gamxxz = l_Gamxxz;
    // double l_Gamxyy = l_Gamxyy; double l_Gamxyz = l_Gamxyz; double l_Gamxzz = l_Gamxzz;
    // double l_Gamyxx = l_Gamyxx; double l_Gamyxy = l_Gamyxy; double l_Gamyxz = l_Gamyxz;
    // double l_Gamyyy = l_Gamyyy; double l_Gamyyz = l_Gamyyz; double l_Gamyzz = l_Gamyzz;
    // double l_Gamzxx = l_Gamzxx; double l_Gamzxy = l_Gamzxy; double l_Gamzxz = l_Gamzxz;
    // double l_Gamzyy = l_Gamzyy; double l_Gamzyz = l_Gamzyz; double l_Gamzzz = l_Gamzzz;

    // double gxxx = gxxx_in[idx]; double gxxy = gxxy_in[idx]; double gxxz = gxxz_in[idx];
    // double gxyx = gxyx_in[idx]; double gxyy = gxyy_in[idx]; double gxzy = gxzy_in[idx]; // 注意: Fortran代码中命名不一致，这里对应 gxy_z
    // double gxzx = gxzx_in[idx]; double gxzz = gxzz_in[idx]; 

    // double gxyz = gxyz_in[idx]; double gyyx = gyyx_in[idx]; double gyyy = gyyy_in[idx];
    // double gyyz = gyyz_in[idx]; double gyzx = gyzx_in[idx]; double gyzy = gyzy_in[idx];
    // double gyzz = gyzz_in[idx]; double gzzx = gzzx_in[idx]; double gzzy = gzzy_in[idx];
    // double gzzz = gzzz_in[idx];

    // Ricci storage starts directly from the batched metric Hessians below.
    // Aij and shift-dependent terms are deliberately loaded after Ricci so
    // they are not live through the connection expansion.
    double l_Rxx, l_Rxy, l_Rxz, l_Ryy, l_Ryz, l_Rzz;

    double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz +
                   TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz +
                   TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz +
                   TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);
    
    double dGamxx, dGamxy, dGamxz;
    double dGamyx, dGamyy, dGamyz;
    double dGamzx, dGamzy, dGamzz;
    d_fderivs_point(dims, Gamx, &dGamxx, &dGamxy, &dGamxz, X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Gamy, &dGamyx, &dGamyy, &dGamyz, X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, Gamz, &dGamzx, &dGamzy, &dGamzz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

    // ==========================================
    // Step 3: Ricci (Metric 二阶导数部分)
    // ==========================================
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
    
    // The six metric Hessian contractions are produced by the preceding
    // variable-batched stencil kernel on this stream.
    l_Rxx = Rxx[idx]; l_Rxy = Rxy[idx]; l_Rxz = Rxz[idx];
    l_Ryy = Ryy[idx]; l_Ryz = Ryz[idx]; l_Rzz = Rzz[idx];

    // ==========================================
    // Step 4: Ricci (连接系数项) - 完整展开
    // ==========================================

    // double Gam_dot_dg_xx = Gamxa * gxxx + Gamya * gxyx + Gamza * gxzx;
    // double Gam_dot_dg_yy = Gamxa * gxyy + Gamya * gyyy + Gamza * gyzy;
    // double Gam_dot_dg_zz = Gamxa * gxzz + Gamya * gyzz + Gamza * gzzz;

    // Rxx Correction
    l_Rxx = -HALF * l_Rxx + 
          l_gxx * dGamxx + l_gxy * dGamyx + l_gxz * dGamzx + 
          Gamxa * gxxx + Gamya * gxyx + Gamza * gxzx + 
          gupxx * (TWO*(l_Gamxxx*gxxx + l_Gamyxx*gxyx + l_Gamzxx*gxzx) + l_Gamxxx*gxxx + l_Gamyxx*gxxy + l_Gamzxx*gxxz) +
          gupxy * (TWO*(l_Gamxxx*gxyx + l_Gamyxx*gyyx + l_Gamzxx*gyzx + l_Gamxxy*gxxx + l_Gamyxy*gxyx + l_Gamzxy*gxzx) + l_Gamxxy*gxxx + l_Gamyxy*gxxy + l_Gamzxy*gxxz + l_Gamxxx*gxyx + l_Gamyxx*gxyy + l_Gamzxx*gxyz) + 
          gupxz * (TWO*(l_Gamxxx*gxzx + l_Gamyxx*gyzx + l_Gamzxx*gzzx + l_Gamxxz*gxxx + l_Gamyxz*gxyx + l_Gamzxz*gxzx) + l_Gamxxz*gxxx + l_Gamyxz*gxxy + l_Gamzxz*gxxz + l_Gamxxx*gxzx + l_Gamyxx*gxzy + l_Gamzxx*gxzz) + 
          gupyy * (TWO*(l_Gamxxy*gxyx + l_Gamyxy*gyyx + l_Gamzxy*gyzx) + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz) + 
          gupyz * (TWO*(l_Gamxxy*gxzx + l_Gamyxy*gyzx + l_Gamzxy*gzzx + l_Gamxxz*gxyx + l_Gamyxz*gyyx + l_Gamzxz*gyzx) + l_Gamxxz*gxyx + l_Gamyxz*gxyy + l_Gamzxz*gxyz + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz) + 
          gupzz * (TWO*(l_Gamxxz*gxzx + l_Gamyxz*gyzx + l_Gamzxz*gzzx) + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz);

    // Ryy Correction
    l_Ryy = -HALF * l_Ryy + 
          l_gxy * dGamxy + l_gyy * dGamyy + l_gyz * dGamzy + 
          Gamxa * gxyy + Gamya * gyyy + Gamza * gyzy + 
          gupxx * (TWO*(l_Gamxxy*gxxy + l_Gamyxy*gxyy + l_Gamzxy*gxzy) + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz) + 
          gupxy * (TWO*(l_Gamxxy*gxyy + l_Gamyxy*gyyy + l_Gamzxy*gyzy + l_Gamxyy*gxxy + l_Gamyyy*gxyy + l_Gamzyy*gxzy) + l_Gamxyy*gxyx + l_Gamyyy*gxyy + l_Gamzyy*gxyz + l_Gamxxy*gyyx + l_Gamyxy*gyyy + l_Gamzxy*gyyz) + 
          gupxz * (TWO*(l_Gamxxy*gxzy + l_Gamyxy*gyzy + l_Gamzxy*gzzy + l_Gamxyz*gxxy + l_Gamyyz*gxyy + l_Gamzyz*gxzy) + l_Gamxyz*gxyx + l_Gamyyz*gxyy + l_Gamzyz*gxyz + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) + 
          gupyy * (TWO*(l_Gamxyy*gxyy + l_Gamyyy*gyyy + l_Gamzyy*gyzy) + l_Gamxyy*gyyx + l_Gamyyy*gyyy + l_Gamzyy*gyyz) + 
          gupyz * (TWO*(l_Gamxyy*gxzy + l_Gamyyy*gyzy + l_Gamzyy*gzzy + l_Gamxyz*gxyy + l_Gamyyz*gyyy + l_Gamzyz*gyzy) + l_Gamxyz*gyyx + l_Gamyyz*gyyy + l_Gamzyz*gyyz + l_Gamxyy*gyzx + l_Gamyyy*gyzy + l_Gamzyy*gyzz) + 
          gupzz * (TWO*(l_Gamxyz*gxzy + l_Gamyyz*gyzy + l_Gamzyz*gzzy) + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz);

    // Rzz Correction
    l_Rzz = -HALF * l_Rzz + 
          l_gxz * dGamxz + l_gyz * dGamyz + l_gzz * dGamzz + 
          Gamxa * gxzz + Gamya * gyzz + Gamza * gzzz + 
          gupxx * (TWO*(l_Gamxxz*gxxz + l_Gamyxz*gxyz + l_Gamzxz*gxzz) + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz) + 
          gupxy * (TWO*(l_Gamxxz*gxyz + l_Gamyxz*gyyz + l_Gamzxz*gyzz + l_Gamxyz*gxxz + l_Gamyyz*gxyz + l_Gamzyz*gxzz) + l_Gamxyz*gxzx + l_Gamyyz*gxzy + l_Gamzyz*gxzz + l_Gamxxz*gyzx + l_Gamyxz*gyzy + l_Gamzxz*gyzz) + 
          gupxz * (TWO*(l_Gamxxz*gxzz + l_Gamyxz*gyzz + l_Gamzxz*gzzz + l_Gamxzz*gxxz + l_Gamyzz*gxyz + l_Gamzzz*gxzz) + l_Gamxzz*gxzx + l_Gamyzz*gxzy + l_Gamzzz*gxzz + l_Gamxxz*gzzx + l_Gamyxz*gzzy + l_Gamzxz*gzzz) + 
          gupyy * (TWO*(l_Gamxyz*gxyz + l_Gamyyz*gyyz + l_Gamzyz*gyzz) + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz) + 
          gupyz * (TWO*(l_Gamxyz*gxzz + l_Gamyyz*gyzz + l_Gamzyz*gzzz + l_Gamxzz*gxyz + l_Gamyzz*gyyz + l_Gamzzz*gyzz) + l_Gamxzz*gyzx + l_Gamyzz*gyzy + l_Gamzzz*gyzz + l_Gamxyz*gzzx + l_Gamyyz*gzzy + l_Gamzyz*gzzz) + 
          gupzz * (TWO*(l_Gamxzz*gxzz + l_Gamyzz*gyzz + l_Gamzzz*gzzz) + l_Gamxzz*gzzx + l_Gamyzz*gzzy + l_Gamzzz*gzzz);

    // Rxy Correction
    l_Rxy = HALF * ( - l_Rxy + 
          l_gxx * dGamxy + l_gxy * dGamyy + l_gxz * dGamzy + 
          l_gxy * dGamxx + l_gyy * dGamyx + l_gyz * dGamzx + 
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
          l_gxx * dGamxz + l_gxy * dGamyz + l_gxz * dGamzz + 
          l_gxz * dGamxx + l_gyz * dGamyx + l_gzz * dGamzx + 
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
          l_gxy * dGamxz + l_gyy * dGamyz + l_gyz * dGamzz + 
          l_gxz * dGamxy + l_gyz * dGamyy + l_gzz * dGamzy + 
          Gamxa * gxzy + Gamya * gyzy + Gamza * gzzy + 
          Gamxa * gxyz + Gamya * gyyz + Gamza * gyzz) + 
          gupxx * (l_Gamxxy*gxxz + l_Gamyxy*gxyz + l_Gamzxy*gxzz + l_Gamxxz*gxxy + l_Gamyxz*gxyy + l_Gamzxz*gxzy + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz) + 
          gupxy * (l_Gamxxy*gxyz + l_Gamyxy*gyyz + l_Gamzxy*gyzz + l_Gamxxz*gxyy + l_Gamyxz*gyyy + l_Gamzxz*gyzy + l_Gamxyy*gxzx + l_Gamyyy*gxzy + l_Gamzyy*gxzz + l_Gamxyy*gxxz + l_Gamyyy*gxyz + l_Gamzyy*gxzz + l_Gamxyz*gxxy + l_Gamyyz*gxyy + l_Gamzyz*gxzy + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) + 
          gupxz * (l_Gamxxy*gxzz + l_Gamyxy*gyzz + l_Gamzxy*gzzz + l_Gamxxz*gxzy + l_Gamyxz*gyzy + l_Gamzxz*gzzy + l_Gamxyz*gxzx + l_Gamyyz*gxzy + l_Gamzyz*gxzz + l_Gamxyz*gxxz + l_Gamyyz*gxyz + l_Gamzyz*gxzz + l_Gamxzz*gxxy + l_Gamyzz*gxyy + l_Gamzzz*gxzy + l_Gamxxy*gzzx + l_Gamyxy*gzzy + l_Gamzxy*gzzz) + 
          gupyy * (l_Gamxyy*gxyz + l_Gamyyy*gyyz + l_Gamzyy*gyzz + l_Gamxyz*gxyy + l_Gamyyz*gyyy + l_Gamzyz*gyzy + l_Gamxyy*gyzx + l_Gamyyy*gyzy + l_Gamzyy*gyzz) + 
          gupyz * (l_Gamxyy*gxzz + l_Gamyyy*gyzz + l_Gamzyy*gzzz + l_Gamxyz*gxzy + l_Gamyyz*gyzy + l_Gamzyz*gzzy + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz + l_Gamxyz*gxyz + l_Gamyyz*gyyz + l_Gamzyz*gyzz + l_Gamxzz*gxyy + l_Gamyzz*gyyy + l_Gamzzz*gyzy + l_Gamxyy*gzzx + l_Gamyyy*gzzy + l_Gamzyy*gzzz) + 
          gupzz * (l_Gamxyz*gxzz + l_Gamyyz*gyzz + l_Gamzyz*gzzz + l_Gamxzz*gxzy + l_Gamyzz*gyzy + l_Gamzzz*gzzy + l_Gamxyz*gzzx + l_Gamyyz*gzzy + l_Gamzyz*gzzz);

    // ==========================================
    // Step 6: Chi 二阶导数与 Ricci 修正
    // ==========================================
    double fxx, fxy, fxz, fyy, fyz, fzz;
    d_fdderivs_point(dims, chi, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    
    // 协变导数修正
    fxx -= l_Gamxxx * chix + l_Gamyxx * chiy + l_Gamzxx * chiz;
    fxy -= l_Gamxxy * chix + l_Gamyxy * chiy + l_Gamzxy * chiz;
    fxz -= l_Gamxxz * chix + l_Gamyxz * chiy + l_Gamzxz * chiz;
    fyy -= l_Gamxyy * chix + l_Gamyyy * chiy + l_Gamzyy * chiz;
    fyz -= l_Gamxyz * chix + l_Gamyyz * chiy + l_Gamzyz * chiz;
    fzz -= l_Gamxzz * chix + l_Gamyzz * chiy + l_Gamzzz * chiz;

    double f_scalar = gupxx * (fxx - F3o2/chin1 * chix * chix) + 
                      gupyy * (fyy - F3o2/chin1 * chiy * chiy) + 
                      gupzz * (fzz - F3o2/chin1 * chiz * chiz) + 
                      TWO * (gupxy * (fxy - F3o2/chin1 * chix * chiy) + 
                             gupxz * (fxz - F3o2/chin1 * chix * chiz) + 
                             gupyz * (fyz - F3o2/chin1 * chiy * chiz));
    
    // Add to Ricci
    l_Rxx += (fxx - chix*chix/chin1/TWO + l_gxx * f_scalar)/chin1/TWO;
    l_Ryy += (fyy - chiy*chiy/chin1/TWO + l_gyy * f_scalar)/chin1/TWO;
    l_Rzz += (fzz - chiz*chiz/chin1/TWO + l_gzz * f_scalar)/chin1/TWO;
    l_Rxy += (fxy - chix*chiy/chin1/TWO + l_gxy * f_scalar)/chin1/TWO;
    l_Rxz += (fxz - chix*chiz/chin1/TWO + l_gxz * f_scalar)/chin1/TWO;
    l_Ryz += (fyz - chiy*chiz/chin1/TWO + l_gyz * f_scalar)/chin1/TWO;

    // The equation-only values start here, after all Ricci connection terms
    // have consumed the metric derivatives.  This ordering keeps Aij, beta
    // derivatives, shift Hessians, and Gamma sources out of the Ricci peak.
    const double l_Axx = Axx[idx], l_Axy = Axy[idx], l_Axz = Axz[idx];
    const double l_Ayy = Ayy[idx], l_Ayz = Ayz[idx], l_Azz = Azz[idx];

    double betaxx, betaxy, betaxz;
    double betayx, betayy, betayz;
    double betazx, betazy, betazz;
    d_fderivs_point(dims, betax, &betaxx, &betaxy, &betaxz,
                    X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betay, &betayx, &betayy, &betayz,
                    X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, betaz, &betazx, &betazy, &betazz,
                    X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);
    const double div_beta = betaxx + betayy + betazz;

    chi_rhs[idx] = F2o3 * chin1 * (alpn1 * val_trK - div_beta);
    gxx_rhs[idx] = -TWO * alpn1 * l_Axx - F2o3 * val_gxx * div_beta
                 + TWO * (val_gxx * betaxx + val_gxy * betayx + val_gxz * betazx);
    gyy_rhs[idx] = -TWO * alpn1 * l_Ayy - F2o3 * val_gyy * div_beta
                 + TWO * (val_gxy * betaxy + val_gyy * betayy + val_gyz * betazy);
    gzz_rhs[idx] = -TWO * alpn1 * l_Azz - F2o3 * val_gzz * div_beta
                 + TWO * (val_gxz * betaxz + val_gyz * betayz + val_gzz * betazz);
    gxy_rhs[idx] = -TWO * alpn1 * l_Axy + F1o3 * val_gxy * div_beta
                 + val_gxx * betaxy + val_gxz * betazy
                 + val_gyy * betayx + val_gyz * betazx - val_gxy * betazz;
    gyz_rhs[idx] = -TWO * alpn1 * l_Ayz + F1o3 * val_gyz * div_beta
                 + val_gxy * betaxz + val_gyy * betayz
                 + val_gxz * betaxy + val_gzz * betazy - val_gyz * betaxx;
    gxz_rhs[idx] = -TWO * alpn1 * l_Axz + F1o3 * val_gxz * div_beta
                 + val_gxx * betaxz + val_gxy * betayz
                 + val_gyz * betayx + val_gzz * betazx - val_gxz * betayy;

    double Lapx, Lapy, Lapz, Kx, Ky, Kz;
    d_fderivs_point(dims, Lap, &Lapx, &Lapy, &Lapz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    d_fderivs_point(dims, trK, &Kx, &Ky, &Kz,
                    X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    const double Aupxx = gupxx * gupxx * l_Axx + gupxy * gupxy * l_Ayy + gupxz * gupxz * l_Azz
        + TWO * (gupxx * gupxy * l_Axy + gupxx * gupxz * l_Axz + gupxy * gupxz * l_Ayz);
    const double Aupyy = gupxy * gupxy * l_Axx + gupyy * gupyy * l_Ayy + gupyz * gupyz * l_Azz
        + TWO * (gupxy * gupyy * l_Axy + gupxy * gupyz * l_Axz + gupyy * gupyz * l_Ayz);
    const double Aupzz = gupxz * gupxz * l_Axx + gupyz * gupyz * l_Ayy + gupzz * gupzz * l_Azz
        + TWO * (gupxz * gupyz * l_Axy + gupxz * gupzz * l_Axz + gupyz * gupzz * l_Ayz);
    const double Aupxy = gupxx * gupxy * l_Axx + gupxy * gupyy * l_Ayy + gupxz * gupyz * l_Azz
        + (gupxx * gupyy + gupxy * gupxy) * l_Axy
        + (gupxx * gupyz + gupxz * gupxy) * l_Axz
        + (gupxy * gupyz + gupxz * gupyy) * l_Ayz;
    const double Aupxz = gupxx * gupxz * l_Axx + gupxy * gupyz * l_Ayy + gupxz * gupzz * l_Azz
        + (gupxx * gupyz + gupxy * gupxz) * l_Axy
        + (gupxx * gupzz + gupxz * gupxz) * l_Axz
        + (gupxy * gupzz + gupxz * gupyz) * l_Ayz;
    const double Aupyz = gupxy * gupxz * l_Axx + gupyy * gupyz * l_Ayy + gupyz * gupzz * l_Azz
        + (gupxy * gupyz + gupyy * gupxz) * l_Axy
        + (gupxy * gupzz + gupyz * gupxz) * l_Axz
        + (gupyy * gupzz + gupyz * gupyz) * l_Ayz;

    const double val_Sx = Sx[idx], val_Sy = Sy[idx], val_Sz = Sz[idx];
    double val_Gamx_rhs = -TWO * (Lapx * Aupxx + Lapy * Aupxy + Lapz * Aupxz)
        + TWO * alpn1 * (
            -F3o2 / chin1 * (chix * Aupxx + chiy * Aupxy + chiz * Aupxz)
            - gupxx * (F2o3 * Kx + EIGHT * PI * val_Sx)
            - gupxy * (F2o3 * Ky + EIGHT * PI * val_Sy)
            - gupxz * (F2o3 * Kz + EIGHT * PI * val_Sz)
            + l_Gamxxx * Aupxx + l_Gamxyy * Aupyy + l_Gamxzz * Aupzz
            + TWO * (l_Gamxxy * Aupxy + l_Gamxxz * Aupxz + l_Gamxyz * Aupyz));
    double val_Gamy_rhs = -TWO * (Lapx * Aupxy + Lapy * Aupyy + Lapz * Aupyz)
        + TWO * alpn1 * (
            -F3o2 / chin1 * (chix * Aupxy + chiy * Aupyy + chiz * Aupyz)
            - gupxy * (F2o3 * Kx + EIGHT * PI * val_Sx)
            - gupyy * (F2o3 * Ky + EIGHT * PI * val_Sy)
            - gupyz * (F2o3 * Kz + EIGHT * PI * val_Sz)
            + l_Gamyxx * Aupxx + l_Gamyyy * Aupyy + l_Gamyzz * Aupzz
            + TWO * (l_Gamyxy * Aupxy + l_Gamyxz * Aupxz + l_Gamyyz * Aupyz));
    double val_Gamz_rhs = -TWO * (Lapx * Aupxz + Lapy * Aupyz + Lapz * Aupzz)
        + TWO * alpn1 * (
            -F3o2 / chin1 * (chix * Aupxz + chiy * Aupyz + chiz * Aupzz)
            - gupxz * (F2o3 * Kx + EIGHT * PI * val_Sx)
            - gupyz * (F2o3 * Ky + EIGHT * PI * val_Sy)
            - gupzz * (F2o3 * Kz + EIGHT * PI * val_Sz)
            + l_Gamzxx * Aupxx + l_Gamzyy * Aupyy + l_Gamzzz * Aupzz
            + TWO * (l_Gamzxy * Aupxy + l_Gamzxz * Aupxz + l_Gamzyz * Aupyz));

    const double bx_gxxx = Gamxxx[idx], bx_gxyx = Gamxxy[idx];
    const double bx_gxzx = Gamxxz[idx], bx_gyyx = Gamxyy[idx];
    const double bx_gyzx = Gamxyz[idx], bx_gzzx = Gamxzz[idx];
    const double by_gxxy = Gamyxx[idx], by_gxyy = Gamyxy[idx];
    const double by_gxzy = Gamyxz[idx], by_gyyy = Gamyyy[idx];
    const double by_gyzy = Gamyyz[idx], by_gzzy = Gamyzz[idx];
    const double bz_gxxz = Gamzxx[idx], bz_gxyz = Gamzxy[idx];
    const double bz_gxzz = Gamzxz[idx], bz_gyyz = Gamzyy[idx];
    const double bz_gyzz = Gamzyz[idx], bz_gzzz = Gamzzz[idx];
    const double shift_trace_x = bx_gxxx + by_gxyy + bz_gxzz;
    const double shift_trace_y = bx_gxyx + by_gyyy + bz_gyzz;
    const double shift_trace_z = bx_gxzx + by_gyzy + bz_gzzz;

    val_Gamx_rhs += F2o3 * Gamxa * div_beta
        - (Gamxa * betaxx + Gamya * betaxy + Gamza * betaxz)
        + F1o3 * (gupxx * shift_trace_x + gupxy * shift_trace_y + gupxz * shift_trace_z)
        + gupxx * bx_gxxx + gupyy * bx_gyyx + gupzz * bx_gzzx
        + TWO * (gupxy * bx_gxyx + gupxz * bx_gxzx + gupyz * bx_gyzx);
    val_Gamy_rhs += F2o3 * Gamya * div_beta
        - (Gamxa * betayx + Gamya * betayy + Gamza * betayz)
        + F1o3 * (gupxy * shift_trace_x + gupyy * shift_trace_y + gupyz * shift_trace_z)
        + gupxx * by_gxxy + gupyy * by_gyyy + gupzz * by_gzzy
        + TWO * (gupxy * by_gxyy + gupxz * by_gxzy + gupyz * by_gyzy);
    val_Gamz_rhs += F2o3 * Gamza * div_beta
        - (Gamxa * betazx + Gamya * betazy + Gamza * betazz)
        + F1o3 * (gupxz * shift_trace_x + gupyz * shift_trace_y + gupzz * shift_trace_z)
        + gupxx * bz_gxxz + gupyy * bz_gyyz + gupzz * bz_gzzz
        + TWO * (gupxy * bz_gxyz + gupxz * bz_gxzz + gupyz * bz_gyzz);

    // ==========================================
    // Step 7: Lapse 二阶导数 & trK_rhs
    // ==========================================
    d_fdderivs_point(dims, Lap, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    // 计算物理连接系数 (暂存到 Gam 数组中以节省寄存器，最后会写回 Global)
    double gx_phy = (gupxx * chix + gupxy * chiy + gupxz * chiz)/chin1;
    double gy_phy = (gupxy * chix + gupyy * chiy + gupyz * chiz)/chin1;
    double gz_phy = (gupxz * chix + gupyz * chiy + gupzz * chiz)/chin1;
    
    // 更新为物理连接系数 (对应 Fortran 241-258)
    l_Gamxxx -= ((chix + chix)/chin1 - l_gxx * gx_phy)*HALF; // l_Gamxxx = l_Gamxxx;
    l_Gamyxx -= (                            - l_gxx * gy_phy)*HALF; // l_Gamyxx = l_Gamyxx;
    l_Gamzxx -= (                            - l_gxx * gz_phy)*HALF; // l_Gamzxx = l_Gamzxx;
    l_Gamxyy -= (                            - l_gyy * gx_phy)*HALF; // l_Gamxyy = l_Gamxyy;
    l_Gamyyy -= ((chiy + chiy)/chin1 - l_gyy * gy_phy)*HALF; // l_Gamyyy = l_Gamyyy;
    l_Gamzyy -= (                            - l_gyy * gz_phy)*HALF; // l_Gamzyy = l_Gamzyy;
    l_Gamxzz -= (                            - l_gzz * gx_phy)*HALF; // l_Gamxzz = l_Gamxzz;
    l_Gamyzz -= (                            - l_gzz * gy_phy)*HALF; // l_Gamyzz = l_Gamyzz;
    l_Gamzzz -= ((chiz + chiz)/chin1 - l_gzz * gz_phy)*HALF; // l_Gamzzz = l_Gamzzz;

    l_Gamxxy -= (chiy/chin1 - l_gxy * gx_phy)*HALF; // l_Gamxxy = l_Gamxxy;
    l_Gamyxy -= (chix/chin1 - l_gxy * gy_phy)*HALF; // l_Gamyxy = l_Gamyxy;
    l_Gamzxy -= (                 - l_gxy * gz_phy)*HALF; // l_Gamzxy = l_Gamzxy;
    l_Gamxxz -= (chiz/chin1 - l_gxz * gx_phy)*HALF; // l_Gamxxz = l_Gamxxz;
    l_Gamyxz -= (                 - l_gxz * gy_phy)*HALF; // l_Gamyxz = l_Gamyxz;
    l_Gamzxz -= (chix/chin1 - l_gxz * gz_phy)*HALF; // l_Gamzxz = l_Gamzxz;
    l_Gamxyz -= (                 - l_gyz * gx_phy)*HALF; // l_Gamxyz = l_Gamxyz;
    l_Gamyyz -= (chiz/chin1 - l_gyz * gy_phy)*HALF; // l_Gamyyz = l_Gamyyz;
    l_Gamzyz -= (chiy/chin1 - l_gyz * gz_phy)*HALF; // l_Gamzyz = l_Gamzyz;

    // Lapse 的协变导数 D_i D_j alpha
    fxx = fxx - l_Gamxxx*Lapx - l_Gamyxx*Lapy - l_Gamzxx*Lapz;
    fyy = fyy - l_Gamxyy*Lapx - l_Gamyyy*Lapy - l_Gamzyy*Lapz;
    fzz = fzz - l_Gamxzz*Lapx - l_Gamyzz*Lapy - l_Gamzzz*Lapz;
    fxy = fxy - l_Gamxxy*Lapx - l_Gamyxy*Lapy - l_Gamzxy*Lapz;
    fxz = fxz - l_Gamxxz*Lapx - l_Gamyxz*Lapy - l_Gamzxz*Lapz;
    fyz = fyz - l_Gamxyz*Lapx - l_Gamyyz*Lapy - l_Gamzyz*Lapz;

    double trK_rhs_val = gupxx * fxx + gupyy * fyy + gupzz * fzz + TWO* (gupxy * fxy + gupxz * fxz + gupyz * fyz);

    // ==========================================
    // Step 8: 组装 Aij_rhs & trK_rhs
    // ==========================================
    double S = chin1 * (gupxx * Sxx[idx] + gupyy * Syy[idx] + gupzz * Szz[idx] + 
               TWO * (gupxy * Sxy[idx] + gupxz * Sxz[idx] + gupyz * Syz[idx]));

    double term_xx = gupxx * l_Axx * l_Axx + gupyy * l_Axy * l_Axy + gupzz * l_Axz * l_Axz + TWO * (gupxy * l_Axx * l_Axy + gupxz * l_Axx * l_Axz + gupyz * l_Axy * l_Axz);
    double term_yy = gupxx * l_Axy * l_Axy + gupyy * l_Ayy * l_Ayy + gupzz * l_Ayz * l_Ayz + TWO * (gupxy * l_Axy * l_Ayy + gupxz * l_Axy * l_Ayz + gupyz * l_Ayy * l_Ayz);
    double term_zz = gupxx * l_Axz * l_Axz + gupyy * l_Ayz * l_Ayz + gupzz * l_Azz * l_Azz + TWO * (gupxy * l_Axz * l_Ayz + gupxz * l_Axz * l_Azz + gupyz * l_Ayz * l_Azz);
    double term_xy = gupxx * l_Axx * l_Axy + gupyy * l_Axy * l_Ayy + gupzz * l_Axz * l_Ayz + gupxy * (l_Axx * l_Ayy + l_Axy * l_Axy) + gupxz * (l_Axx * l_Ayz + l_Axz * l_Axy) + gupyz * (l_Axy * l_Ayz + l_Axz * l_Ayy);
    double term_xz = gupxx * l_Axx * l_Axz + gupyy * l_Axy * l_Ayz + gupzz * l_Axz * l_Azz + gupxy * (l_Axx * l_Ayz + l_Axy * l_Axz) + gupxz * (l_Axx * l_Azz + l_Axz * l_Axz) + gupyz * (l_Axy * l_Azz + l_Axz * l_Ayz);
    double term_yz = gupxx * l_Axy * l_Axz + gupyy * l_Ayy * l_Ayz + gupzz * l_Ayz * l_Azz + gupxy * (l_Axy * l_Ayz + l_Ayy * l_Axz) + gupxz * (l_Axy * l_Azz + l_Ayz * l_Axz) + gupyz * (l_Ayy * l_Azz + l_Ayz * l_Ayz);

    double trA2 = gupxx * term_xx + gupyy * term_yy + gupzz * term_zz + TWO * (gupxy * term_xy + gupxz * term_xz + gupyz * term_yz);

    double f = F2o3 * val_trK * val_trK - trA2 - F16*PI*rho[idx] + EIGHT*PI*S;
    double f_trace = -F1o3 * (trK_rhs_val + alpn1/chin1 * f);

    // 计算 Aij 源项
    double src_xx = alpn1 * (l_Rxx - EIGHT*PI*Sxx[idx]) - fxx; // fxx is D_i D_j Lap
    double src_yy = alpn1 * (l_Ryy - EIGHT*PI*Syy[idx]) - fyy;
    double src_zz = alpn1 * (l_Rzz - EIGHT*PI*Szz[idx]) - fzz;
    double src_xy = alpn1 * (l_Rxy - EIGHT*PI*Sxy[idx]) - fxy;
    double src_xz = alpn1 * (l_Rxz - EIGHT*PI*Sxz[idx]) - fxz;
    double src_yz = alpn1 * (l_Ryz - EIGHT*PI*Syz[idx]) - fyz;

    double Axx_rhs_val = src_xx - l_gxx * f_trace;
    double Ayy_rhs_val = src_yy - l_gyy * f_trace;
    double Azz_rhs_val = src_zz - l_gzz * f_trace;
    double Axy_rhs_val = src_xy - l_gxy * f_trace;
    double Axz_rhs_val = src_xz - l_gxz * f_trace;
    double Ayz_rhs_val = src_yz - l_gyz * f_trace;

    // 添加平流项 (Lie derivative of Aij)
    Axx_rhs[idx] = chin1 * Axx_rhs_val + alpn1 * (val_trK * l_Axx - TWO * term_xx) + TWO * (l_Axx * betaxx + l_Axy * betayx + l_Axz * betazx) - F2o3 * l_Axx * div_beta;
    Ayy_rhs[idx] = chin1 * Ayy_rhs_val + alpn1 * (val_trK * l_Ayy - TWO * term_yy) + TWO * (l_Axy * betaxy + l_Ayy * betayy + l_Ayz * betazy) - F2o3 * l_Ayy * div_beta;
    Azz_rhs[idx] = chin1 * Azz_rhs_val + alpn1 * (val_trK * l_Azz - TWO * term_zz) + TWO * (l_Axz * betaxz + l_Ayz * betayz + l_Azz * betazz) - F2o3 * l_Azz * div_beta;
    
    Axy_rhs[idx] = chin1 * Axy_rhs_val + alpn1 * (val_trK * l_Axy - TWO * term_xy) + l_Axx * betaxy + l_Axz * betazy + l_Ayy * betayx + l_Ayz * betazx - l_Axy * betazz + F1o3 * l_Axy * div_beta;
    Ayz_rhs[idx] = chin1 * Ayz_rhs_val + alpn1 * (val_trK * l_Ayz - TWO * term_yz) + l_Axy * betaxz + l_Ayy * betayz + l_Axz * betaxy + l_Azz * betazy - l_Ayz * betaxx + F1o3 * l_Ayz * div_beta;
    Axz_rhs[idx] = chin1 * Axz_rhs_val + alpn1 * (val_trK * l_Axz - TWO * term_xz) + l_Axx * betaxz + l_Axy * betayz + l_Ayz * betayx + l_Azz * betazx - l_Axz * betayy + F1o3 * l_Axz * div_beta;

    trK_rhs[idx] = -chin1 * trK_rhs_val + alpn1 * (F1o3 * val_trK * val_trK + trA2 + FOUR * PI * (rho[idx] + S));

    // Gauge vars RHS
    Lap_rhs[idx] = -TWO * alpn1 * val_trK;
    betax_rhs[idx] = FF * dtSfx[idx];
    betay_rhs[idx] = FF * dtSfy[idx];
    betaz_rhs[idx] = FF * dtSfz[idx];
    dtSfx_rhs[idx] = val_Gamx_rhs - eta * dtSfx[idx];
    dtSfy_rhs[idx] = val_Gamy_rhs - eta * dtSfy[idx];
    dtSfz_rhs[idx] = val_Gamz_rhs - eta * dtSfz[idx];

    Gamx_rhs[idx] = val_Gamx_rhs;
    Gamy_rhs[idx] = val_Gamy_rhs;
    Gamz_rhs[idx] = val_Gamz_rhs;

    // Only the predictor computes constraints. Materialize its geometry for the
    // following constraint kernels; corrector stages avoid these 24 stores.
    if (co == 0) {
        Gamxxx[idx] = l_Gamxxx; Gamxxy[idx] = l_Gamxxy; Gamxxz[idx] = l_Gamxxz;
        Gamxyy[idx] = l_Gamxyy; Gamxyz[idx] = l_Gamxyz; Gamxzz[idx] = l_Gamxzz;
        Gamyxx[idx] = l_Gamyxx; Gamyxy[idx] = l_Gamyxy; Gamyxz[idx] = l_Gamyxz;
        Gamyyy[idx] = l_Gamyyy; Gamyyz[idx] = l_Gamyyz; Gamyzz[idx] = l_Gamyzz;
        Gamzxx[idx] = l_Gamzxx; Gamzxy[idx] = l_Gamzxy; Gamzxz[idx] = l_Gamzxz;
        Gamzyy[idx] = l_Gamzyy; Gamzyz[idx] = l_Gamzyz; Gamzzz[idx] = l_Gamzzz;
        Rxx[idx] = l_Rxx; Rxy[idx] = l_Rxy; Rxz[idx] = l_Rxz;
        Ryy[idx] = l_Ryy; Ryz[idx] = l_Ryz; Rzz[idx] = l_Rzz;
    }

    return;

}

struct RhsAdvectionBatch {
    const double* field[24];
    double* rhs[24];
    signed char parity[24][3];
};

struct InverseMetric {
    double xx, xy, xz, yy, yz, zz;
};

__device__ __forceinline__ InverseMetric inverse_metric(
    double gxx, double gxy, double gxz,
    double gyy, double gyz, double gzz
) {
    const double det =
        gxx * gyy * gzz + TWO * gxy * gyz * gxz
        - gxz * gyy * gxz - gxy * gxy * gzz - gxx * gyz * gyz;
    InverseMetric inv;
    inv.xx = (gyy * gzz - gyz * gyz) / det;
    inv.xy = -(gxy * gzz - gyz * gxz) / det;
    inv.xz = (gxy * gyz - gyy * gxz) / det;
    inv.yy = (gxx * gzz - gxz * gxz) / det;
    inv.yz = -(gxx * gyz - gxy * gxz) / det;
    inv.zz = (gxx * gyy - gxy * gxy) / det;
    return inv;
}

struct MetricHessianBatch {
    const double* field[6];
    double* contraction[6];
    signed char parity[6][3];
};

struct ShiftHessianBatch {
    const double* field[3];
    double* derivative[3][6];
    signed char parity[3][3];
};

__global__ void rhs_shift_hessian_batch_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    ShiftHessianBatch batch, int symmetry, int lev
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int points = ex0 * ex1 * ex2;
    if (idx >= points) return;

    const int i = idx % ex0;
    const int plane_idx = idx / ex0;
    const int j = plane_idx % ex1;
    const int k = plane_idx / ex1;
    const int dims[3] = {ex0, ex1, ex2};
    const int variable = blockIdx.y;

    double fxx, fxy, fxz, fyy, fyz, fzz;
    const double sx = static_cast<double>(batch.parity[variable][0]);
    const double sy = static_cast<double>(batch.parity[variable][1]);
    const double sz = static_cast<double>(batch.parity[variable][2]);
    d_fdderivs_point(
        dims, batch.field[variable],
        &fxx, &fxy, &fxz, &fyy, &fyz, &fzz,
        X, Y, Z, sx, sy, sz, symmetry, lev, i, j, k
    );
    batch.derivative[variable][0][idx] = fxx;
    batch.derivative[variable][1][idx] = fxy;
    batch.derivative[variable][2][idx] = fxz;
    batch.derivative[variable][3][idx] = fyy;
    batch.derivative[variable][4][idx] = fyz;
    batch.derivative[variable][5][idx] = fzz;
}

__global__ void rhs_metric_hessian_batch_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* dxx, const double* gxy, const double* gxz,
    const double* dyy, const double* gyz, const double* dzz,
    MetricHessianBatch batch, int symmetry, int lev
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int points = ex0 * ex1 * ex2;
    if (idx >= points) return;

    const int i = idx % ex0;
    const int plane_idx = idx / ex0;
    const int j = plane_idx % ex1;
    const int k = plane_idx / ex1;
    const int dims[3] = {ex0, ex1, ex2};
    const InverseMetric inv = inverse_metric(
        dxx[idx] + ONE, gxy[idx], gxz[idx],
        dyy[idx] + ONE, gyz[idx], dzz[idx] + ONE
    );

#pragma unroll 1
    for (int offset = 0; offset < 2; ++offset) {
        const int variable = blockIdx.y * 2 + offset;
        double fxx, fxy, fxz, fyy, fyz, fzz;
        const double sx = static_cast<double>(batch.parity[variable][0]);
        const double sy = static_cast<double>(batch.parity[variable][1]);
        const double sz = static_cast<double>(batch.parity[variable][2]);
        d_fdderivs_point(
            dims, batch.field[variable],
            &fxx, &fxy, &fxz, &fyy, &fyz, &fzz,
            X, Y, Z, sx, sy, sz, symmetry, lev, i, j, k
        );
        batch.contraction[variable][idx] =
            inv.xx * fxx + inv.yy * fyy + inv.zz * fzz
            + TWO * (inv.xy * fxy + inv.xz * fxz + inv.yz * fyz);
    }
}

__global__ void rhs_advection_ko_batch_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* betax, const double* betay, const double* betaz,
    RhsAdvectionBatch batch, int symmetry, double eps
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int points = ex0 * ex1 * ex2;
    if (idx >= points) return;

    const int i = idx % ex0;
    const int plane_idx = idx / ex0;
    const int j = plane_idx % ex1;
    const int k = plane_idx / ex1;
    const int dims[3] = {ex0, ex1, ex2};

    // Four fields per thread amortizes point/velocity setup while leaving six
    // independent variable groups in the grid for the 14-SM MIG partition.
#pragma unroll 1
    for (int offset = 0; offset < 4; ++offset) {
        const int variable = blockIdx.y * 4 + offset;
        const double* field = batch.field[variable];
        double* rhs = batch.rhs[variable];
        const double sx = static_cast<double>(batch.parity[variable][0]);
        const double sy = static_cast<double>(batch.parity[variable][1]);
        const double sz = static_cast<double>(batch.parity[variable][2]);

        double value = rhs[idx];
        value += d_lopsided_point(
            dims, field, rhs, betax, betay, betaz, X, Y, Z,
            symmetry, sx, sy, sz, i, j, k
        );
        if (eps > 0.0) {
            value += d_kodis_point(
                dims, field, X, Y, Z, sx, sy, sz,
                symmetry, eps, i, j, k
            );
        }
        rhs[idx] = value;
    }
}

__global__ void rhs_hamiltonian_constraint_kernel(
    int ex0, int ex1, int ex2,
    const double* chi, const double* trK,
    const double* dxx, const double* gxy, const double* gxz,
    const double* dyy, const double* gyz, const double* dzz,
    const double* Axx, const double* Axy, const double* Axz,
    const double* Ayy, const double* Ayz, const double* Azz,
    const double* Rxx, const double* Rxy, const double* Rxz,
    const double* Ryy, const double* Ryz, const double* Rzz,
    const double* rho, double* ham_Res
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);

    const InverseMetric inv = inverse_metric(
        dxx[idx] + ONE, gxy[idx], gxz[idx],
        dyy[idx] + ONE, gyz[idx], dzz[idx] + ONE
    );
    const double axx = Axx[idx];
    const double axy = Axy[idx];
    const double axz = Axz[idx];
    const double ayy = Ayy[idx];
    const double ayz = Ayz[idx];
    const double azz = Azz[idx];

    const double term_xx =
        inv.xx * axx * axx + inv.yy * axy * axy + inv.zz * axz * axz
        + TWO * (inv.xy * axx * axy + inv.xz * axx * axz + inv.yz * axy * axz);
    const double term_yy =
        inv.xx * axy * axy + inv.yy * ayy * ayy + inv.zz * ayz * ayz
        + TWO * (inv.xy * axy * ayy + inv.xz * axy * ayz + inv.yz * ayy * ayz);
    const double term_zz =
        inv.xx * axz * axz + inv.yy * ayz * ayz + inv.zz * azz * azz
        + TWO * (inv.xy * axz * ayz + inv.xz * axz * azz + inv.yz * ayz * azz);
    const double term_xy =
        inv.xx * axx * axy + inv.yy * axy * ayy + inv.zz * axz * ayz
        + inv.xy * (axx * ayy + axy * axy)
        + inv.xz * (axx * ayz + axz * axy)
        + inv.yz * (axy * ayz + axz * ayy);
    const double term_xz =
        inv.xx * axx * axz + inv.yy * axy * ayz + inv.zz * axz * azz
        + inv.xy * (axx * ayz + axy * axz)
        + inv.xz * (axx * azz + axz * axz)
        + inv.yz * (axy * azz + axz * ayz);
    const double term_yz =
        inv.xx * axy * axz + inv.yy * ayy * ayz + inv.zz * ayz * azz
        + inv.xy * (axy * ayz + ayy * axz)
        + inv.xz * (axy * azz + ayz * axz)
        + inv.yz * (ayy * azz + ayz * ayz);
    const double trA2 =
        inv.xx * term_xx + inv.yy * term_yy + inv.zz * term_zz
        + TWO * (inv.xy * term_xy + inv.xz * term_xz + inv.yz * term_yz);
    const double trR =
        inv.xx * Rxx[idx] + inv.yy * Ryy[idx] + inv.zz * Rzz[idx]
        + TWO * (inv.xy * Rxy[idx] + inv.xz * Rxz[idx] + inv.yz * Ryz[idx]);
    const double K = trK[idx];
    ham_Res[idx] = (chi[idx] + ONE) * trR + F2o3 * K * K - trA2 - F16 * PI * rho[idx];
}

struct SymmetricTensor {
    double xx, xy, xz, yy, yz, zz;
};

struct Christoffel {
    double xxx, xxy, xxz, xyy, xyz, xzz;
    double yxx, yxy, yxz, yyy, yyz, yzz;
    double zxx, zxy, zxz, zyy, zyz, zzz;
};

template<int I, int J>
__device__ __forceinline__ double tensor_component(const SymmetricTensor& a) {
    if ((I == 0) && (J == 0)) return a.xx;
    if (((I == 0) && (J == 1)) || ((I == 1) && (J == 0))) return a.xy;
    if (((I == 0) && (J == 2)) || ((I == 2) && (J == 0))) return a.xz;
    if ((I == 1) && (J == 1)) return a.yy;
    if (((I == 1) && (J == 2)) || ((I == 2) && (J == 1))) return a.yz;
    return a.zz;
}

template<int U, int I, int J>
__device__ __forceinline__ double christoffel_component(const Christoffel& g) {
    if (U == 0) {
        if ((I == 0) && (J == 0)) return g.xxx;
        if (((I == 0) && (J == 1)) || ((I == 1) && (J == 0))) return g.xxy;
        if (((I == 0) && (J == 2)) || ((I == 2) && (J == 0))) return g.xxz;
        if ((I == 1) && (J == 1)) return g.xyy;
        if (((I == 1) && (J == 2)) || ((I == 2) && (J == 1))) return g.xyz;
        return g.xzz;
    }
    if (U == 1) {
        if ((I == 0) && (J == 0)) return g.yxx;
        if (((I == 0) && (J == 1)) || ((I == 1) && (J == 0))) return g.yxy;
        if (((I == 0) && (J == 2)) || ((I == 2) && (J == 0))) return g.yxz;
        if ((I == 1) && (J == 1)) return g.yyy;
        if (((I == 1) && (J == 2)) || ((I == 2) && (J == 1))) return g.yyz;
        return g.yzz;
    }
    if ((I == 0) && (J == 0)) return g.zxx;
    if (((I == 0) && (J == 1)) || ((I == 1) && (J == 0))) return g.zxy;
    if (((I == 0) && (J == 2)) || ((I == 2) && (J == 0))) return g.zxz;
    if ((I == 1) && (J == 1)) return g.zyy;
    if (((I == 1) && (J == 2)) || ((I == 2) && (J == 1))) return g.zyz;
    return g.zzz;
}

template<int DIR, int P, int Q>
__device__ __forceinline__ double covariant_A_derivative(
    double partial, const SymmetricTensor& a, const Christoffel& g,
    double chi_derivative, double chin1
) {
    return partial
        - christoffel_component<0, P, DIR>(g) * tensor_component<0, Q>(a)
        - christoffel_component<1, P, DIR>(g) * tensor_component<1, Q>(a)
        - christoffel_component<2, P, DIR>(g) * tensor_component<2, Q>(a)
        - christoffel_component<0, Q, DIR>(g) * tensor_component<P, 0>(a)
        - christoffel_component<1, Q, DIR>(g) * tensor_component<P, 1>(a)
        - christoffel_component<2, Q, DIR>(g) * tensor_component<P, 2>(a)
        - chi_derivative * tensor_component<P, Q>(a) / chin1;
}

template<int J>
__global__ void rhs_momentum_constraint_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* chi, const double* trK,
    const double* dxx, const double* gxy, const double* gxz,
    const double* dyy, const double* gyz, const double* dzz,
    const double* Axx, const double* Axy, const double* Axz,
    const double* Ayy, const double* Ayz, const double* Azz,
    const double* Gamxxx, const double* Gamxxy, const double* Gamxxz,
    const double* Gamxyy, const double* Gamxyz, const double* Gamxzz,
    const double* Gamyxx, const double* Gamyxy, const double* Gamyxz,
    const double* Gamyyy, const double* Gamyyz, const double* Gamyzz,
    const double* Gamzxx, const double* Gamzxy, const double* Gamzxz,
    const double* Gamzyy, const double* Gamzyz, const double* Gamzzz,
    const double* matter_momentum, double* momentum_residual,
    int symmetry, int lev
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    const int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    const int dims[3] = {ex0, ex1, ex2};

    const InverseMetric inv = inverse_metric(
        dxx[idx] + ONE, gxy[idx], gxz[idx],
        dyy[idx] + ONE, gyz[idx], dzz[idx] + ONE
    );
    const SymmetricTensor a = {
        Axx[idx], Axy[idx], Axz[idx], Ayy[idx], Ayz[idx], Azz[idx]
    };
    const Christoffel g = {
        Gamxxx[idx], Gamxxy[idx], Gamxxz[idx],
        Gamxyy[idx], Gamxyz[idx], Gamxzz[idx],
        Gamyxx[idx], Gamyxy[idx], Gamyxz[idx],
        Gamyyy[idx], Gamyyz[idx], Gamyzz[idx],
        Gamzxx[idx], Gamzxy[idx], Gamzxz[idx],
        Gamzyy[idx], Gamzyz[idx], Gamzzz[idx]
    };

    double chix, chiy, chiz;
    d_fderivs_point(
        dims, chi, &chix, &chiy, &chiz, X, Y, Z,
        SYM, SYM, SYM, symmetry, lev, i, j, k
    );

    const double* field0 = (J == 0) ? Axx : ((J == 1) ? Axy : Axz);
    const double* field1 = (J == 0) ? Axy : ((J == 1) ? Ayy : Ayz);
    const double* field2 = (J == 0) ? Axz : ((J == 1) ? Ayz : Azz);

    double dx, dy, dz;
    d_fderivs_point(
        dims, field0, &dx, &dy, &dz, X, Y, Z,
        (J == 0) ? SYM : ANTI,
        (J == 1) ? ANTI : SYM,
        (J == 2) ? ANTI : SYM,
        symmetry, lev, i, j, k
    );
    double residual =
        inv.xx * covariant_A_derivative<0, 0, J>(dx, a, g, chix, chi[idx] + ONE)
        + inv.xy * covariant_A_derivative<1, 0, J>(dy, a, g, chiy, chi[idx] + ONE)
        + inv.xz * covariant_A_derivative<2, 0, J>(dz, a, g, chiz, chi[idx] + ONE);

    d_fderivs_point(
        dims, field1, &dx, &dy, &dz, X, Y, Z,
        (J == 0) ? ANTI : SYM,
        (J == 1) ? SYM : ANTI,
        (J == 2) ? ANTI : SYM,
        symmetry, lev, i, j, k
    );
    residual +=
        inv.xy * covariant_A_derivative<0, 1, J>(dx, a, g, chix, chi[idx] + ONE)
        + inv.yy * covariant_A_derivative<1, 1, J>(dy, a, g, chiy, chi[idx] + ONE)
        + inv.yz * covariant_A_derivative<2, 1, J>(dz, a, g, chiz, chi[idx] + ONE);

    d_fderivs_point(
        dims, field2, &dx, &dy, &dz, X, Y, Z,
        (J == 0) ? ANTI : SYM,
        (J == 1) ? ANTI : SYM,
        (J == 2) ? SYM : ANTI,
        symmetry, lev, i, j, k
    );
    residual +=
        inv.xz * covariant_A_derivative<0, 2, J>(dx, a, g, chix, chi[idx] + ONE)
        + inv.yz * covariant_A_derivative<1, 2, J>(dy, a, g, chiy, chi[idx] + ONE)
        + inv.zz * covariant_A_derivative<2, 2, J>(dz, a, g, chiz, chi[idx] + ONE);

    double Kx, Ky, Kz;
    d_fderivs_point(
        dims, trK, &Kx, &Ky, &Kz, X, Y, Z,
        SYM, SYM, SYM, symmetry, lev, i, j, k
    );
    const double K_derivative = (J == 0) ? Kx : ((J == 1) ? Ky : Kz);
    momentum_residual[idx] =
        residual - F2o3 * K_derivative - F8 * PI * matter_momentum[idx];
}

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
    dim3 block(8, 8, 4); // 调整 block size 以适应架构
    dim3 grid(
        (ex[0] + block.x - 1) / block.x,
        (ex[1] + block.y - 1) / block.y,
        (ex[2] + block.z - 1) / block.z
    );
    const int points = ex[0] * ex[1] * ex[2];

    const MetricHessianBatch metric_hessian_batch = {
        {d_dxx, d_dyy, d_dzz, d_gxy, d_gxz, d_gyz},
        {d_Rxx, d_Ryy, d_Rzz, d_Rxy, d_Rxz, d_Ryz},
        {
            { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1},
            {-1, -1,  1}, {-1,  1, -1}, { 1, -1, -1}
        }
    };
    const int stencil_threads = 256;
    const dim3 metric_hessian_grid(
        (points + stencil_threads - 1) / stencil_threads, 3, 1
    );
    rhs_metric_hessian_batch_kernel<<<
        metric_hessian_grid, stencil_threads, 0, stream
    >>>(
        ex[0], ex[1], ex[2], d_X, d_Y, d_Z,
        d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
        metric_hessian_batch, symmetry, lev
    );

    const ShiftHessianBatch shift_hessian_batch = {
        {d_betax, d_betay, d_betaz},
        {
            {d_Gamxxx, d_Gamxxy, d_Gamxxz, d_Gamxyy, d_Gamxyz, d_Gamxzz},
            {d_Gamyxx, d_Gamyxy, d_Gamyxz, d_Gamyyy, d_Gamyyz, d_Gamyzz},
            {d_Gamzxx, d_Gamzxy, d_Gamzxz, d_Gamzyy, d_Gamzyz, d_Gamzzz}
        },
        {{-1, 1, 1}, {1, -1, 1}, {1, 1, -1}}
    };
    const dim3 shift_hessian_grid(
        (points + stencil_threads - 1) / stencil_threads, 3, 1
    );
    rhs_shift_hessian_batch_kernel<<<
        shift_hessian_grid, stencil_threads, 0, stream
    >>>(
        ex[0], ex[1], ex[2], d_X, d_Y, d_Z,
        shift_hessian_batch, symmetry, lev
    );

    rhs_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], T, d_X, d_Y, d_Z,
        d_chi, d_trK,
        d_dxx, d_gxy, d_gxz,
        d_dyy, d_gyz, d_dzz,
        d_Axx, d_Axy, d_Axz,
        d_Ayy, d_Ayz, d_Azz,
        d_Gamx, d_Gamy, d_Gamz,
        d_Lap,
        d_betax, d_betay, d_betaz,
        d_dtSfx, d_dtSfy, d_dtSfz,
        d_chi_rhs, d_trK_rhs,
        d_gxx_rhs, d_gxy_rhs, d_gxz_rhs,
        d_gyy_rhs, d_gyz_rhs, d_gzz_rhs,
        d_Axx_rhs, d_Axy_rhs, d_Axz_rhs,
        d_Ayy_rhs, d_Ayz_rhs, d_Azz_rhs,
        d_Gamx_rhs, d_Gamy_rhs, d_Gamz_rhs,
        d_Lap_rhs,
        d_betax_rhs, d_betay_rhs, d_betaz_rhs,
        d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs,
        d_rho, d_Sx, d_Sy, d_Sz,
        d_Sxx, d_Sxy, d_Sxz,
        d_Syy, d_Syz, d_Szz,
        d_Gamxxx, d_Gamxxy, d_Gamxxz,
        d_Gamxyy, d_Gamxyz, d_Gamxzz,
        d_Gamyxx, d_Gamyxy, d_Gamyxz,
        d_Gamyyy, d_Gamyyz, d_Gamyzz,
        d_Gamzxx, d_Gamzxy, d_Gamzxz,
        d_Gamzyy, d_Gamzyz, d_Gamzzz,
        d_Rxx, d_Rxy, d_Rxz,
        d_Ryy, d_Ryz, d_Rzz,
        d_ham_Res, d_movx_Res, d_movy_Res, d_movz_Res,
        d_Gmx_Res, d_Gmy_Res, d_Gmz_Res,
        symmetry, lev, eps, co
    );

    const RhsAdvectionBatch advection_batch = {
        {
            d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
            d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
            d_chi, d_trK, d_Gamx, d_Gamy, d_Gamz, d_Lap,
            d_betax, d_betay, d_betaz, d_dtSfx, d_dtSfy, d_dtSfz
        },
        {
            d_gxx_rhs, d_gxy_rhs, d_gxz_rhs, d_gyy_rhs, d_gyz_rhs, d_gzz_rhs,
            d_Axx_rhs, d_Axy_rhs, d_Axz_rhs, d_Ayy_rhs, d_Ayz_rhs, d_Azz_rhs,
            d_chi_rhs, d_trK_rhs, d_Gamx_rhs, d_Gamy_rhs, d_Gamz_rhs, d_Lap_rhs,
            d_betax_rhs, d_betay_rhs, d_betaz_rhs,
            d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs
        },
        {
            { 1,  1,  1}, {-1, -1,  1}, {-1,  1, -1},
            { 1,  1,  1}, { 1, -1, -1}, { 1,  1,  1},
            { 1,  1,  1}, {-1, -1,  1}, {-1,  1, -1},
            { 1,  1,  1}, { 1, -1, -1}, { 1,  1,  1},
            { 1,  1,  1}, { 1,  1,  1},
            {-1,  1,  1}, { 1, -1,  1}, { 1,  1, -1},
            { 1,  1,  1},
            {-1,  1,  1}, { 1, -1,  1}, { 1,  1, -1},
            {-1,  1,  1}, { 1, -1,  1}, { 1,  1, -1}
        }
    };
    const int advection_threads = 256;
    const dim3 advection_grid(
        (points + advection_threads - 1) / advection_threads, 6, 1
    );
    rhs_advection_ko_batch_kernel<<<advection_grid, advection_threads, 0, stream>>>(
        ex[0], ex[1], ex[2], d_X, d_Y, d_Z,
        d_betax, d_betay, d_betaz, advection_batch, symmetry, eps
    );

    if (co == 0) {
        rhs_hamiltonian_constraint_kernel<<<grid, block, 0, stream>>>(
            ex[0], ex[1], ex[2],
            d_chi, d_trK,
            d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
            d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
            d_Rxx, d_Rxy, d_Rxz, d_Ryy, d_Ryz, d_Rzz,
            d_rho, d_ham_Res
        );

        rhs_momentum_constraint_kernel<0><<<grid, block, 0, stream>>>(
            ex[0], ex[1], ex[2], d_X, d_Y, d_Z, d_chi, d_trK,
            d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
            d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
            d_Gamxxx, d_Gamxxy, d_Gamxxz, d_Gamxyy, d_Gamxyz, d_Gamxzz,
            d_Gamyxx, d_Gamyxy, d_Gamyxz, d_Gamyyy, d_Gamyyz, d_Gamyzz,
            d_Gamzxx, d_Gamzxy, d_Gamzxz, d_Gamzyy, d_Gamzyz, d_Gamzzz,
            d_Sx, d_movx_Res, symmetry, lev
        );
        rhs_momentum_constraint_kernel<1><<<grid, block, 0, stream>>>(
            ex[0], ex[1], ex[2], d_X, d_Y, d_Z, d_chi, d_trK,
            d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
            d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
            d_Gamxxx, d_Gamxxy, d_Gamxxz, d_Gamxyy, d_Gamxyz, d_Gamxzz,
            d_Gamyxx, d_Gamyxy, d_Gamyxz, d_Gamyyy, d_Gamyyz, d_Gamyzz,
            d_Gamzxx, d_Gamzxy, d_Gamzxz, d_Gamzyy, d_Gamzyz, d_Gamzzz,
            d_Sy, d_movy_Res, symmetry, lev
        );
        rhs_momentum_constraint_kernel<2><<<grid, block, 0, stream>>>(
            ex[0], ex[1], ex[2], d_X, d_Y, d_Z, d_chi, d_trK,
            d_dxx, d_gxy, d_gxz, d_dyy, d_gyz, d_dzz,
            d_Axx, d_Axy, d_Axz, d_Ayy, d_Ayz, d_Azz,
            d_Gamxxx, d_Gamxxy, d_Gamxxz, d_Gamxyy, d_Gamxyz, d_Gamxzz,
            d_Gamyxx, d_Gamyxy, d_Gamyxz, d_Gamyyy, d_Gamyyz, d_Gamyzz,
            d_Gamzxx, d_Gamzxy, d_Gamzxz, d_Gamzyy, d_Gamzyz, d_Gamzzz,
            d_Sz, d_movz_Res, symmetry, lev
        );
    }
}
