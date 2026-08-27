#ifndef HESSIAN_COMPACT_GPU_CUH
#define HESSIAN_COMPACT_GPU_CUH

#include <cuda_runtime.h>
#include <math.h>

constexpr int COMPACT_HESSIAN_FIELDS = 6;
constexpr int COMPACT_HESSIAN_RADIUS = 2;
constexpr int COMPACT_HESSIAN_BX = 8;
constexpr int COMPACT_HESSIAN_BY = 8;
constexpr int COMPACT_HESSIAN_BZ = 4;
constexpr int COMPACT_HESSIAN_SX =
    COMPACT_HESSIAN_BX + 2 * COMPACT_HESSIAN_RADIUS;
constexpr int COMPACT_HESSIAN_SY =
    COMPACT_HESSIAN_BY + 2 * COMPACT_HESSIAN_RADIUS;
constexpr int COMPACT_HESSIAN_SZ =
    COMPACT_HESSIAN_BZ + 2 * COMPACT_HESSIAN_RADIUS;
constexpr int COMPACT_HESSIAN_PITCH = COMPACT_HESSIAN_SX + 1;
constexpr int COMPACT_HESSIAN_LOGICAL_SIZE =
    COMPACT_HESSIAN_SX * COMPACT_HESSIAN_SY * COMPACT_HESSIAN_SZ;
constexpr int COMPACT_HESSIAN_TILE_SIZE =
    COMPACT_HESSIAN_PITCH * COMPACT_HESSIAN_SY * COMPACT_HESSIAN_SZ;

struct CompactHessianFields {
    const double* input[COMPACT_HESSIAN_FIELDS];
    double* output[COMPACT_HESSIAN_FIELDS];
    int parity_x[COMPACT_HESSIAN_FIELDS];
    int parity_y[COMPACT_HESSIAN_FIELDS];
    int parity_z[COMPACT_HESSIAN_FIELDS];
};

__device__ __forceinline__ int compact_hessian_tile_index(
    int x, int y, int z
) {
    return x + COMPACT_HESSIAN_PITCH * (y + COMPACT_HESSIAN_SY * z);
}

__device__ __forceinline__ double compact_hessian_symmetry_load(
    const double* field,
    int i, int j, int k,
    int ex0, int ex1, int ex2,
    int parity_x, int parity_y, int parity_z
) {
    if (i < -COMPACT_HESSIAN_RADIUS || i >= ex0 ||
        j < -COMPACT_HESSIAN_RADIUS || j >= ex1 ||
        k < -COMPACT_HESSIAN_RADIUS || k >= ex2) {
        return 0.0;
    }

    double factor = 1.0;
    if (i < 0) {
        i = -i - 1;
        factor *= parity_x;
    }
    if (j < 0) {
        j = -j - 1;
        factor *= parity_y;
    }
    if (k < 0) {
        k = -k - 1;
        factor *= parity_z;
    }
    if (i >= ex0 || j >= ex1 || k >= ex2) return 0.0;
    return factor * field[i + ex0 * (j + ex1 * k)];
}

__device__ __forceinline__ double compact_hessian_tile_at(
    const double* tile, int center_index, int di, int dj, int dk
) {
    return tile[center_index + di + COMPACT_HESSIAN_PITCH *
        (dj + COMPACT_HESSIAN_SY * dk)];
}

__device__ __forceinline__ void compact_hessian_derivatives_from_tile(
    const double* tile, int center_index,
    bool active, int i, int j, int k,
    int ex0, int ex1, int ex2, int kmin,
    const double* scales,
    double& fxx, double& fxy, double& fxz,
    double& fyy, double& fyz, double& fzz
) {
    fxx = 0.0;
    fxy = 0.0;
    fxz = 0.0;
    fyy = 0.0;
    fyz = 0.0;
    fzz = 0.0;

#define HESS_AT(di, dj, dk) \
    compact_hessian_tile_at(tile, center_index, (di), (dj), (dk))
    if (active &&
        i + 2 <= ex0 - 1 && i - 2 >= 0 &&
        j + 2 <= ex1 - 1 && j - 2 >= 0 &&
        k + 2 <= ex2 - 1 && k - 2 >= kmin) {
        fxx = scales[3] * (-HESS_AT(-2, 0, 0) + 16.0 * HESS_AT(-1, 0, 0)
            - 30.0 * HESS_AT(0, 0, 0) - HESS_AT(2, 0, 0)
            + 16.0 * HESS_AT(1, 0, 0));
        fyy = scales[4] * (-HESS_AT(0, -2, 0) + 16.0 * HESS_AT(0, -1, 0)
            - 30.0 * HESS_AT(0, 0, 0) - HESS_AT(0, 2, 0)
            + 16.0 * HESS_AT(0, 1, 0));
        fzz = scales[5] * (-HESS_AT(0, 0, -2) + 16.0 * HESS_AT(0, 0, -1)
            - 30.0 * HESS_AT(0, 0, 0) - HESS_AT(0, 0, 2)
            + 16.0 * HESS_AT(0, 0, 1));

        fxy = scales[9] * (
            (HESS_AT(-2, -2, 0) - 8.0 * HESS_AT(-1, -2, 0)
                + 8.0 * HESS_AT(1, -2, 0) - HESS_AT(2, -2, 0))
            - 8.0 * (HESS_AT(-2, -1, 0) - 8.0 * HESS_AT(-1, -1, 0)
                + 8.0 * HESS_AT(1, -1, 0) - HESS_AT(2, -1, 0))
            + 8.0 * (HESS_AT(-2, 1, 0) - 8.0 * HESS_AT(-1, 1, 0)
                + 8.0 * HESS_AT(1, 1, 0) - HESS_AT(2, 1, 0))
            - (HESS_AT(-2, 2, 0) - 8.0 * HESS_AT(-1, 2, 0)
                + 8.0 * HESS_AT(1, 2, 0) - HESS_AT(2, 2, 0)));
        fxz = scales[10] * (
            (HESS_AT(-2, 0, -2) - 8.0 * HESS_AT(-1, 0, -2)
                + 8.0 * HESS_AT(1, 0, -2) - HESS_AT(2, 0, -2))
            - 8.0 * (HESS_AT(-2, 0, -1) - 8.0 * HESS_AT(-1, 0, -1)
                + 8.0 * HESS_AT(1, 0, -1) - HESS_AT(2, 0, -1))
            + 8.0 * (HESS_AT(-2, 0, 1) - 8.0 * HESS_AT(-1, 0, 1)
                + 8.0 * HESS_AT(1, 0, 1) - HESS_AT(2, 0, 1))
            - (HESS_AT(-2, 0, 2) - 8.0 * HESS_AT(-1, 0, 2)
                + 8.0 * HESS_AT(1, 0, 2) - HESS_AT(2, 0, 2)));
        fyz = scales[11] * (
            (HESS_AT(0, -2, -2) - 8.0 * HESS_AT(0, -1, -2)
                + 8.0 * HESS_AT(0, 1, -2) - HESS_AT(0, 2, -2))
            - 8.0 * (HESS_AT(0, -2, -1) - 8.0 * HESS_AT(0, -1, -1)
                + 8.0 * HESS_AT(0, 1, -1) - HESS_AT(0, 2, -1))
            + 8.0 * (HESS_AT(0, -2, 1) - 8.0 * HESS_AT(0, -1, 1)
                + 8.0 * HESS_AT(0, 1, 1) - HESS_AT(0, 2, 1))
            - (HESS_AT(0, -2, 2) - 8.0 * HESS_AT(0, -1, 2)
                + 8.0 * HESS_AT(0, 1, 2) - HESS_AT(0, 2, 2)));
    } else if (active &&
        i + 1 <= ex0 - 1 && i - 1 >= 0 &&
        j + 1 <= ex1 - 1 && j - 1 >= 0 &&
        k + 1 <= ex2 - 1 && k - 1 >= kmin) {
        fxx = scales[0] * (HESS_AT(-1, 0, 0)
            - 2.0 * HESS_AT(0, 0, 0) + HESS_AT(1, 0, 0));
        fyy = scales[1] * (HESS_AT(0, -1, 0)
            - 2.0 * HESS_AT(0, 0, 0) + HESS_AT(0, 1, 0));
        fzz = scales[2] * (HESS_AT(0, 0, -1)
            - 2.0 * HESS_AT(0, 0, 0) + HESS_AT(0, 0, 1));
        fxy = scales[6] * (HESS_AT(-1, -1, 0) - HESS_AT(1, -1, 0)
            - HESS_AT(-1, 1, 0) + HESS_AT(1, 1, 0));
        fxz = scales[7] * (HESS_AT(-1, 0, -1) - HESS_AT(1, 0, -1)
            - HESS_AT(-1, 0, 1) + HESS_AT(1, 0, 1));
        fyz = scales[8] * (HESS_AT(0, -1, -1) - HESS_AT(0, 1, -1)
            - HESS_AT(0, -1, 1) + HESS_AT(0, 1, 1));
    }
#undef HESS_AT
}

__device__ __forceinline__ void compact_first_derivatives_from_tile(
    const double* tile, int center_index,
    bool active, int i, int j, int k,
    int ex0, int ex1, int ex2, int kmin,
    const double* scales,
    double& fx, double& fy, double& fz
) {
    fx = 0.0;
    fy = 0.0;
    fz = 0.0;

#define FIRST_AT(di, dj, dk) \
    compact_hessian_tile_at(tile, center_index, (di), (dj), (dk))
    if (active &&
        i + 2 <= ex0 - 1 && i - 2 >= 0 &&
        j + 2 <= ex1 - 1 && j - 2 >= 0 &&
        k + 2 <= ex2 - 1 && k - 2 >= kmin) {
        fx = scales[12] * (FIRST_AT(-2, 0, 0) - 8.0 * FIRST_AT(-1, 0, 0)
            + 8.0 * FIRST_AT(1, 0, 0) - FIRST_AT(2, 0, 0));
        fy = scales[13] * (FIRST_AT(0, -2, 0) - 8.0 * FIRST_AT(0, -1, 0)
            + 8.0 * FIRST_AT(0, 1, 0) - FIRST_AT(0, 2, 0));
        fz = scales[14] * (FIRST_AT(0, 0, -2) - 8.0 * FIRST_AT(0, 0, -1)
            + 8.0 * FIRST_AT(0, 0, 1) - FIRST_AT(0, 0, 2));
    } else if (active &&
        i + 1 <= ex0 - 1 && i - 1 >= 0 &&
        j + 1 <= ex1 - 1 && j - 1 >= 0 &&
        k + 1 <= ex2 - 1 && k - 1 >= kmin) {
        fx = scales[15] * (-FIRST_AT(-1, 0, 0) + FIRST_AT(1, 0, 0));
        fy = scales[16] * (-FIRST_AT(0, -1, 0) + FIRST_AT(0, 1, 0));
        fz = scales[17] * (-FIRST_AT(0, 0, -1) + FIRST_AT(0, 0, 1));
    }
#undef FIRST_AT
}

constexpr int COMPACT_GEOMETRY_GRADIENT_FIELDS = 4;
constexpr int COMPACT_GEOMETRY_METRIC_FIELDS = 6;

struct CompactGeometryRicciFields {
    const double* gradient_input[COMPACT_GEOMETRY_GRADIENT_FIELDS];
    double* gradient_output[COMPACT_GEOMETRY_GRADIENT_FIELDS][3];
    int gradient_parity[COMPACT_GEOMETRY_GRADIENT_FIELDS][3];
    const double* metric[COMPACT_GEOMETRY_METRIC_FIELDS];
    int metric_parity[COMPACT_GEOMETRY_METRIC_FIELDS][3];
    const double* a[COMPACT_GEOMETRY_METRIC_FIELDS];
    double* connection[3][COMPACT_GEOMETRY_METRIC_FIELDS];
    double* inverse_metric[COMPACT_GEOMETRY_METRIC_FIELDS];
    double* ricci[COMPACT_GEOMETRY_METRIC_FIELDS];
};

__global__ void rhs_geometry_ricci_a_equatorial_compact_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    CompactGeometryRicciFields fields
) {
    const int base_i = blockIdx.x * COMPACT_HESSIAN_BX;
    const int base_j = blockIdx.y * COMPACT_HESSIAN_BY;
    const int base_k = blockIdx.z * COMPACT_HESSIAN_BZ;
    const bool block_interior =
        base_i >= COMPACT_HESSIAN_RADIUS &&
        base_j >= COMPACT_HESSIAN_RADIUS &&
        base_k >= COMPACT_HESSIAN_RADIUS &&
        base_i + COMPACT_HESSIAN_BX + COMPACT_HESSIAN_RADIUS <= ex0 &&
        base_j + COMPACT_HESSIAN_BY + COMPACT_HESSIAN_RADIUS <= ex1 &&
        base_k + COMPACT_HESSIAN_BZ + COMPACT_HESSIAN_RADIUS <= ex2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tx + COMPACT_HESSIAN_BX *
        (ty + COMPACT_HESSIAN_BY * tz);
    const int threads = COMPACT_HESSIAN_BX *
        COMPACT_HESSIAN_BY * COMPACT_HESSIAN_BZ;
    const int i = base_i + tx;
    const int j = base_j + ty;
    const int k = base_k + tz;
    const bool valid = i < ex0 && j < ex1 && k < ex2;
    const bool active = valid && i < ex0 - 1 && j < ex1 - 1 && k < ex2 - 1;
    const int idx = valid ? i + ex0 * (j + ex1 * k) : 0;
    const int center_index = compact_hessian_tile_index(
        tx + COMPACT_HESSIAN_RADIUS,
        ty + COMPACT_HESSIAN_RADIUS,
        tz + COMPACT_HESSIAN_RADIUS
    );

    __shared__ double tile[COMPACT_HESSIAN_TILE_SIZE];
    __shared__ double scales[18];
    __shared__ int kmin_shared;
    if (tid == 0) {
        const double dx = X[1] - X[0];
        const double dy = Y[1] - Y[0];
        const double dz = Z[1] - Z[0];
        scales[12] = (1.0 / 12.0) / dx;
        scales[13] = (1.0 / 12.0) / dy;
        scales[14] = (1.0 / 12.0) / dz;
        scales[15] = (1.0 / 2.0) / dx;
        scales[16] = (1.0 / 2.0) / dy;
        scales[17] = (1.0 / 2.0) / dz;
        kmin_shared = (fabs(Z[0]) < dz) ? -2 : 0;
    }

#pragma unroll 1
    for (int field_index = 0;
         field_index < COMPACT_GEOMETRY_GRADIENT_FIELDS;
         ++field_index) {
        const double* field = fields.gradient_input[field_index];
        for (int p = tid; p < COMPACT_HESSIAN_LOGICAL_SIZE; p += threads) {
            const int tile_i = p % COMPACT_HESSIAN_SX;
            const int q = p / COMPACT_HESSIAN_SX;
            const int tile_j = q % COMPACT_HESSIAN_SY;
            const int tile_k = q / COMPACT_HESSIAN_SY;
            const int gi = base_i + tile_i - COMPACT_HESSIAN_RADIUS;
            const int gj = base_j + tile_j - COMPACT_HESSIAN_RADIUS;
            const int gk = base_k + tile_k - COMPACT_HESSIAN_RADIUS;
            const int tile_index = compact_hessian_tile_index(
                tile_i, tile_j, tile_k
            );
            tile[tile_index] = block_interior
                ? field[gi + ex0 * (gj + ex1 * gk)]
                : compact_hessian_symmetry_load(
                    field, gi, gj, gk, ex0, ex1, ex2,
                    fields.gradient_parity[field_index][0],
                    fields.gradient_parity[field_index][1],
                    fields.gradient_parity[field_index][2]
                );
        }
        __syncthreads();
        if (valid) {
            double fx, fy, fz;
            compact_first_derivatives_from_tile(
                tile, center_index, active, i, j, k,
                ex0, ex1, ex2, kmin_shared, scales,
                fx, fy, fz
            );
            fields.gradient_output[field_index][0][idx] = fx;
            fields.gradient_output[field_index][1][idx] = fy;
            fields.gradient_output[field_index][2][idx] = fz;
        }
        if (field_index + 1 < COMPACT_GEOMETRY_GRADIENT_FIELDS) {
            __syncthreads();
        }
    }
    __syncthreads();

    double gxxx = 0.0, gxxy = 0.0, gxxz = 0.0;
    double gxyx = 0.0, gxyy = 0.0, gxyz = 0.0;
    double gxzx = 0.0, gxzy = 0.0, gxzz = 0.0;
    double gyyx = 0.0, gyyy = 0.0, gyyz = 0.0;
    double gyzx = 0.0, gyzy = 0.0, gyzz = 0.0;
    double gzzx = 0.0, gzzy = 0.0, gzzz = 0.0;

#pragma unroll 1
    for (int field_index = 0;
         field_index < COMPACT_GEOMETRY_METRIC_FIELDS;
         ++field_index) {
        const double* field = fields.metric[field_index];
        for (int p = tid; p < COMPACT_HESSIAN_LOGICAL_SIZE; p += threads) {
            const int tile_i = p % COMPACT_HESSIAN_SX;
            const int q = p / COMPACT_HESSIAN_SX;
            const int tile_j = q % COMPACT_HESSIAN_SY;
            const int tile_k = q / COMPACT_HESSIAN_SY;
            const int gi = base_i + tile_i - COMPACT_HESSIAN_RADIUS;
            const int gj = base_j + tile_j - COMPACT_HESSIAN_RADIUS;
            const int gk = base_k + tile_k - COMPACT_HESSIAN_RADIUS;
            const int tile_index = compact_hessian_tile_index(
                tile_i, tile_j, tile_k
            );
            tile[tile_index] = block_interior
                ? field[gi + ex0 * (gj + ex1 * gk)]
                : compact_hessian_symmetry_load(
                    field, gi, gj, gk, ex0, ex1, ex2,
                    fields.metric_parity[field_index][0],
                    fields.metric_parity[field_index][1],
                    fields.metric_parity[field_index][2]
                );
        }
        __syncthreads();
        if (valid) {
            double fx, fy, fz;
            compact_first_derivatives_from_tile(
                tile, center_index, active, i, j, k,
                ex0, ex1, ex2, kmin_shared, scales,
                fx, fy, fz
            );
            switch (field_index) {
                case 0: gxxx = fx; gxxy = fy; gxxz = fz; break;
                case 1: gxyx = fx; gxyy = fy; gxyz = fz; break;
                case 2: gxzx = fx; gxzy = fy; gxzz = fz; break;
                case 3: gyyx = fx; gyyy = fy; gyyz = fz; break;
                case 4: gyzx = fx; gyzy = fy; gyzz = fz; break;
                default: gzzx = fx; gzzy = fy; gzzz = fz; break;
            }
        }
        if (field_index + 1 < COMPACT_GEOMETRY_METRIC_FIELDS) {
            __syncthreads();
        }
    }

    if (valid) {
        const double l_gxx = fields.metric[0][idx] + 1.0;
        const double l_gxy = fields.metric[1][idx];
        const double l_gxz = fields.metric[2][idx];
        const double l_gyy = fields.metric[3][idx] + 1.0;
        const double l_gyz = fields.metric[4][idx];
        const double l_gzz = fields.metric[5][idx] + 1.0;
        const double detg = l_gxx * (l_gyy * l_gzz - l_gyz * l_gyz)
                          - l_gxy * (l_gxy * l_gzz - l_gyz * l_gxz)
                          + l_gxz * (l_gxy * l_gyz - l_gyy * l_gxz);
        const double gupxx = (l_gyy * l_gzz - l_gyz * l_gyz) / detg;
        const double gupxy = -(l_gxy * l_gzz - l_gyz * l_gxz) / detg;
        const double gupxz = (l_gxy * l_gyz - l_gyy * l_gxz) / detg;
        const double gupyy = (l_gxx * l_gzz - l_gxz * l_gxz) / detg;
        const double gupyz = -(l_gxx * l_gyz - l_gxy * l_gxz) / detg;
        const double gupzz = (l_gxx * l_gyy - l_gxy * l_gxy) / detg;

        fields.inverse_metric[0][idx] = gupxx;
        fields.inverse_metric[1][idx] = gupxy;
        fields.inverse_metric[2][idx] = gupxz;
        fields.inverse_metric[3][idx] = gupyy;
        fields.inverse_metric[4][idx] = gupyz;
        fields.inverse_metric[5][idx] = gupzz;

        const double l_Axx = fields.a[0][idx];
        const double l_Axy = fields.a[1][idx];
        const double l_Axz = fields.a[2][idx];
        const double l_Ayy = fields.a[3][idx];
        const double l_Ayz = fields.a[4][idx];
        const double l_Azz = fields.a[5][idx];
        fields.ricci[0][idx] =
            gupxx * gupxx * l_Axx + gupxy * gupxy * l_Ayy +
            gupxz * gupxz * l_Azz +
            2.0 * (gupxx * gupxy * l_Axy + gupxx * gupxz * l_Axz +
                   gupxy * gupxz * l_Ayz);
        fields.ricci[3][idx] =
            gupxy * gupxy * l_Axx + gupyy * gupyy * l_Ayy +
            gupyz * gupyz * l_Azz +
            2.0 * (gupxy * gupyy * l_Axy + gupxy * gupyz * l_Axz +
                   gupyy * gupyz * l_Ayz);
        fields.ricci[5][idx] =
            gupxz * gupxz * l_Axx + gupyz * gupyz * l_Ayy +
            gupzz * gupzz * l_Azz +
            2.0 * (gupxz * gupyz * l_Axy + gupxz * gupzz * l_Axz +
                   gupyz * gupzz * l_Ayz);
        fields.ricci[1][idx] =
            gupxx * gupxy * l_Axx + gupxy * gupyy * l_Ayy +
            gupxz * gupyz * l_Azz +
            (gupxx * gupyy + gupxy * gupxy) * l_Axy +
            (gupxx * gupyz + gupxz * gupxy) * l_Axz +
            (gupxy * gupyz + gupxz * gupyy) * l_Ayz;
        fields.ricci[2][idx] =
            gupxx * gupxz * l_Axx + gupxy * gupyz * l_Ayy +
            gupxz * gupzz * l_Azz +
            (gupxx * gupyz + gupxy * gupxz) * l_Axy +
            (gupxx * gupzz + gupxz * gupxz) * l_Axz +
            (gupxy * gupzz + gupxz * gupyz) * l_Ayz;
        fields.ricci[4][idx] =
            gupxy * gupxz * l_Axx + gupyy * gupyz * l_Ayy +
            gupyz * gupzz * l_Azz +
            (gupxy * gupyz + gupyy * gupxz) * l_Axy +
            (gupxy * gupzz + gupyz * gupxz) * l_Axz +
            (gupyy * gupzz + gupyz * gupyz) * l_Ayz;

        fields.connection[0][0][idx] = 0.5 * (
            gupxx * gxxx + gupxy * (2.0 * gxyx - gxxy) +
            gupxz * (2.0 * gxzx - gxxz));
        fields.connection[1][0][idx] = 0.5 * (
            gupxy * gxxx + gupyy * (2.0 * gxyx - gxxy) +
            gupyz * (2.0 * gxzx - gxxz));
        fields.connection[2][0][idx] = 0.5 * (
            gupxz * gxxx + gupyz * (2.0 * gxyx - gxxy) +
            gupzz * (2.0 * gxzx - gxxz));
        fields.connection[0][3][idx] = 0.5 * (
            gupxx * (2.0 * gxyy - gyyx) + gupxy * gyyy +
            gupxz * (2.0 * gyzy - gyyz));
        fields.connection[1][3][idx] = 0.5 * (
            gupxy * (2.0 * gxyy - gyyx) + gupyy * gyyy +
            gupyz * (2.0 * gyzy - gyyz));
        fields.connection[2][3][idx] = 0.5 * (
            gupxz * (2.0 * gxyy - gyyx) + gupyz * gyyy +
            gupzz * (2.0 * gyzy - gyyz));
        fields.connection[0][5][idx] = 0.5 * (
            gupxx * (2.0 * gxzz - gzzx) +
            gupxy * (2.0 * gyzz - gzzy) + gupxz * gzzz);
        fields.connection[1][5][idx] = 0.5 * (
            gupxy * (2.0 * gxzz - gzzx) +
            gupyy * (2.0 * gyzz - gzzy) + gupyz * gzzz);
        fields.connection[2][5][idx] = 0.5 * (
            gupxz * (2.0 * gxzz - gzzx) +
            gupyz * (2.0 * gyzz - gzzy) + gupzz * gzzz);
        fields.connection[0][1][idx] = 0.5 * (
            gupxx * gxxy + gupxy * gyyx +
            gupxz * (gxzy + gyzx - gxyz));
        fields.connection[1][1][idx] = 0.5 * (
            gupxy * gxxy + gupyy * gyyx +
            gupyz * (gxzy + gyzx - gxyz));
        fields.connection[2][1][idx] = 0.5 * (
            gupxz * gxxy + gupyz * gyyx +
            gupzz * (gxzy + gyzx - gxyz));
        fields.connection[0][2][idx] = 0.5 * (
            gupxx * gxxz + gupxy * (gxyz + gyzx - gxzy) +
            gupxz * gzzx);
        fields.connection[1][2][idx] = 0.5 * (
            gupxy * gxxz + gupyy * (gxyz + gyzx - gxzy) +
            gupyz * gzzx);
        fields.connection[2][2][idx] = 0.5 * (
            gupxz * gxxz + gupyz * (gxyz + gyzx - gxzy) +
            gupzz * gzzx);
        fields.connection[0][4][idx] = 0.5 * (
            gupxx * (gxyz + gxzy - gyzx) + gupxy * gyyz +
            gupxz * gzzy);
        fields.connection[1][4][idx] = 0.5 * (
            gupxy * (gxyz + gxzy - gyzx) + gupyy * gyyz +
            gupyz * gzzy);
        fields.connection[2][4][idx] = 0.5 * (
            gupxz * (gxyz + gxzy - gyzx) + gupyz * gyyz +
            gupzz * gzzy);
    }
}

inline void launch_rhs_geometry_ricci_a_equatorial_compact(
    cudaStream_t stream,
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const CompactGeometryRicciFields& fields
) {
    const dim3 block(
        COMPACT_HESSIAN_BX,
        COMPACT_HESSIAN_BY,
        COMPACT_HESSIAN_BZ
    );
    const dim3 grid(
        (ex0 + block.x - 1) / block.x,
        (ex1 + block.y - 1) / block.y,
        (ex2 + block.z - 1) / block.z
    );
    rhs_geometry_ricci_a_equatorial_compact_kernel<<<grid, block, 0, stream>>>(
        ex0, ex1, ex2, X, Y, Z, fields
    );
}

// Compute the same fourth-order Hessian (with the same second-order boundary
// fallback) as d_fdderivs_point, but source all stencil values from one tile.
__global__ void rhs_evolution_equatorial_compact_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* gupxx_field, const double* gupxy_field,
    const double* gupxz_field, const double* gupyy_field,
    const double* gupyz_field, const double* gupzz_field,
    CompactHessianFields fields
) {
    const int base_i = blockIdx.x * COMPACT_HESSIAN_BX;
    const int base_j = blockIdx.y * COMPACT_HESSIAN_BY;
    const int base_k = blockIdx.z * COMPACT_HESSIAN_BZ;
    const bool block_interior =
        base_i >= COMPACT_HESSIAN_RADIUS &&
        base_j >= COMPACT_HESSIAN_RADIUS &&
        base_k >= COMPACT_HESSIAN_RADIUS &&
        base_i + COMPACT_HESSIAN_BX + COMPACT_HESSIAN_RADIUS <= ex0 &&
        base_j + COMPACT_HESSIAN_BY + COMPACT_HESSIAN_RADIUS <= ex1 &&
        base_k + COMPACT_HESSIAN_BZ + COMPACT_HESSIAN_RADIUS <= ex2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tx + COMPACT_HESSIAN_BX *
        (ty + COMPACT_HESSIAN_BY * tz);
    const int threads = COMPACT_HESSIAN_BX *
        COMPACT_HESSIAN_BY * COMPACT_HESSIAN_BZ;
    const int i = base_i + tx;
    const int j = base_j + ty;
    const int k = base_k + tz;
    const bool valid = i < ex0 && j < ex1 && k < ex2;
    const bool active = valid && i < ex0 - 1 && j < ex1 - 1 && k < ex2 - 1;
    const int idx = valid ? i + ex0 * (j + ex1 * k) : 0;

    __shared__ double tile[COMPACT_HESSIAN_TILE_SIZE];
    __shared__ double scales[12];
    __shared__ int kmin_shared;
    if (tid == 0) {
        const double dx = X[1] - X[0];
        const double dy = Y[1] - Y[0];
        const double dz = Z[1] - Z[0];
        scales[0] = 1.0 / (dx * dx);
        scales[1] = 1.0 / (dy * dy);
        scales[2] = 1.0 / (dz * dz);
        scales[3] = (1.0 / 12.0) / (dx * dx);
        scales[4] = (1.0 / 12.0) / (dy * dy);
        scales[5] = (1.0 / 12.0) / (dz * dz);
        scales[6] = 0.25 / (dx * dy);
        scales[7] = 0.25 / (dx * dz);
        scales[8] = 0.25 / (dy * dz);
        scales[9] = (1.0 / 144.0) / (dx * dy);
        scales[10] = (1.0 / 144.0) / (dx * dz);
        scales[11] = (1.0 / 144.0) / (dy * dz);
        kmin_shared = (fabs(Z[0]) < dz) ? -2 : 0;
    }
    __syncthreads();

    const double gupxx = valid ? gupxx_field[idx] : 0.0;
    const double gupxy = valid ? gupxy_field[idx] : 0.0;
    const double gupxz = valid ? gupxz_field[idx] : 0.0;
    const double gupyy = valid ? gupyy_field[idx] : 0.0;
    const double gupyz = valid ? gupyz_field[idx] : 0.0;
    const double gupzz = valid ? gupzz_field[idx] : 0.0;
    const int center_index = compact_hessian_tile_index(
        tx + COMPACT_HESSIAN_RADIUS,
        ty + COMPACT_HESSIAN_RADIUS,
        tz + COMPACT_HESSIAN_RADIUS
    );

#pragma unroll 1
    for (int field_index = 0; field_index < COMPACT_HESSIAN_FIELDS; ++field_index) {
        const double* field = fields.input[field_index];
        for (int p = tid; p < COMPACT_HESSIAN_LOGICAL_SIZE; p += threads) {
            const int tile_i = p % COMPACT_HESSIAN_SX;
            const int q = p / COMPACT_HESSIAN_SX;
            const int tile_j = q % COMPACT_HESSIAN_SY;
            const int tile_k = q / COMPACT_HESSIAN_SY;
            const int gi = base_i + tile_i - COMPACT_HESSIAN_RADIUS;
            const int gj = base_j + tile_j - COMPACT_HESSIAN_RADIUS;
            const int gk = base_k + tile_k - COMPACT_HESSIAN_RADIUS;
            const int tile_index = compact_hessian_tile_index(
                tile_i, tile_j, tile_k
            );
            tile[tile_index] = block_interior
                ? field[gi + ex0 * (gj + ex1 * gk)]
                : compact_hessian_symmetry_load(
                    field, gi, gj, gk, ex0, ex1, ex2,
                    fields.parity_x[field_index],
                    fields.parity_y[field_index],
                    fields.parity_z[field_index]
                );
        }
        __syncthreads();

        if (valid) {
            double fxx, fxy, fxz, fyy, fyz, fzz;
            compact_hessian_derivatives_from_tile(
                tile, center_index, active, i, j, k,
                ex0, ex1, ex2, kmin_shared, scales,
                fxx, fxy, fxz, fyy, fyz, fzz
            );

            fields.output[field_index][idx] =
                gupxx * fxx + gupyy * fyy + gupzz * fzz +
                2.0 * (gupxy * fxy + gupxz * fxz + gupyz * fyz);
        }
        __syncthreads();
    }
}

inline void launch_rhs_evolution_equatorial_compact(
    cudaStream_t stream,
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* gupxx, const double* gupxy, const double* gupxz,
    const double* gupyy, const double* gupyz, const double* gupzz,
    const CompactHessianFields& fields
) {
    const dim3 block(
        COMPACT_HESSIAN_BX,
        COMPACT_HESSIAN_BY,
        COMPACT_HESSIAN_BZ
    );
    const dim3 grid(
        (ex0 + block.x - 1) / block.x,
        (ex1 + block.y - 1) / block.y,
        (ex2 + block.z - 1) / block.z
    );
    rhs_evolution_equatorial_compact_kernel<<<grid, block, 0, stream>>>(
        ex0, ex1, ex2, X, Y, Z,
        gupxx, gupxy, gupxz, gupyy, gupyz, gupzz, fields
    );
}

constexpr int COMPACT_BETA_FIELDS = 3;
constexpr int COMPACT_CONNECTION_COMPONENTS = 6;

struct CompactBetaPrepareFields {
    const double* input[COMPACT_BETA_FIELDS];
    double* divergence[COMPACT_BETA_FIELDS];
    double* laplacian[COMPACT_BETA_FIELDS];
    int parity_x[COMPACT_BETA_FIELDS];
    int parity_y[COMPACT_BETA_FIELDS];
    int parity_z[COMPACT_BETA_FIELDS];
};

struct CompactConnectionFields {
    const double* input[COMPACT_BETA_FIELDS][COMPACT_CONNECTION_COMPONENTS];
    double* contracted[COMPACT_BETA_FIELDS];
};

__global__ void rhs_beta_gamma_prepare_equatorial_compact_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* gupxx_field, const double* gupxy_field,
    const double* gupxz_field, const double* gupyy_field,
    const double* gupyz_field, const double* gupzz_field,
    CompactBetaPrepareFields beta_fields,
    CompactConnectionFields connection_fields
) {
    const int base_i = blockIdx.x * COMPACT_HESSIAN_BX;
    const int base_j = blockIdx.y * COMPACT_HESSIAN_BY;
    const int base_k = blockIdx.z * COMPACT_HESSIAN_BZ;
    const bool block_interior =
        base_i >= COMPACT_HESSIAN_RADIUS &&
        base_j >= COMPACT_HESSIAN_RADIUS &&
        base_k >= COMPACT_HESSIAN_RADIUS &&
        base_i + COMPACT_HESSIAN_BX + COMPACT_HESSIAN_RADIUS <= ex0 &&
        base_j + COMPACT_HESSIAN_BY + COMPACT_HESSIAN_RADIUS <= ex1 &&
        base_k + COMPACT_HESSIAN_BZ + COMPACT_HESSIAN_RADIUS <= ex2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tx + COMPACT_HESSIAN_BX *
        (ty + COMPACT_HESSIAN_BY * tz);
    const int threads = COMPACT_HESSIAN_BX *
        COMPACT_HESSIAN_BY * COMPACT_HESSIAN_BZ;
    const int i = base_i + tx;
    const int j = base_j + ty;
    const int k = base_k + tz;
    const bool valid = i < ex0 && j < ex1 && k < ex2;
    const bool active = valid && i < ex0 - 1 && j < ex1 - 1 && k < ex2 - 1;
    const int idx = valid ? i + ex0 * (j + ex1 * k) : 0;

    __shared__ double tile[COMPACT_HESSIAN_TILE_SIZE];
    __shared__ double scales[12];
    __shared__ int kmin_shared;
    if (tid == 0) {
        const double dx = X[1] - X[0];
        const double dy = Y[1] - Y[0];
        const double dz = Z[1] - Z[0];
        scales[0] = 1.0 / (dx * dx);
        scales[1] = 1.0 / (dy * dy);
        scales[2] = 1.0 / (dz * dz);
        scales[3] = (1.0 / 12.0) / (dx * dx);
        scales[4] = (1.0 / 12.0) / (dy * dy);
        scales[5] = (1.0 / 12.0) / (dz * dz);
        scales[6] = 0.25 / (dx * dy);
        scales[7] = 0.25 / (dx * dz);
        scales[8] = 0.25 / (dy * dz);
        scales[9] = (1.0 / 144.0) / (dx * dy);
        scales[10] = (1.0 / 144.0) / (dx * dz);
        scales[11] = (1.0 / 144.0) / (dy * dz);
        kmin_shared = (fabs(Z[0]) < dz) ? -2 : 0;
    }
    __syncthreads();

    const double gupxx = valid ? gupxx_field[idx] : 0.0;
    const double gupxy = valid ? gupxy_field[idx] : 0.0;
    const double gupxz = valid ? gupxz_field[idx] : 0.0;
    const double gupyy = valid ? gupyy_field[idx] : 0.0;
    const double gupyz = valid ? gupyz_field[idx] : 0.0;
    const double gupzz = valid ? gupzz_field[idx] : 0.0;
    const int center_index = compact_hessian_tile_index(
        tx + COMPACT_HESSIAN_RADIUS,
        ty + COMPACT_HESSIAN_RADIUS,
        tz + COMPACT_HESSIAN_RADIUS
    );
    double div_hess_x = 0.0;
    double div_hess_y = 0.0;
    double div_hess_z = 0.0;

#pragma unroll 1
    for (int field_index = 0; field_index < COMPACT_BETA_FIELDS; ++field_index) {
        const double* field = beta_fields.input[field_index];
        for (int p = tid; p < COMPACT_HESSIAN_LOGICAL_SIZE; p += threads) {
            const int tile_i = p % COMPACT_HESSIAN_SX;
            const int q = p / COMPACT_HESSIAN_SX;
            const int tile_j = q % COMPACT_HESSIAN_SY;
            const int tile_k = q / COMPACT_HESSIAN_SY;
            const int gi = base_i + tile_i - COMPACT_HESSIAN_RADIUS;
            const int gj = base_j + tile_j - COMPACT_HESSIAN_RADIUS;
            const int gk = base_k + tile_k - COMPACT_HESSIAN_RADIUS;
            const int tile_index = compact_hessian_tile_index(
                tile_i, tile_j, tile_k
            );
            tile[tile_index] = block_interior
                ? field[gi + ex0 * (gj + ex1 * gk)]
                : compact_hessian_symmetry_load(
                    field, gi, gj, gk, ex0, ex1, ex2,
                    beta_fields.parity_x[field_index],
                    beta_fields.parity_y[field_index],
                    beta_fields.parity_z[field_index]
                );
        }
        __syncthreads();

        if (valid) {
            double hxx, hxy, hxz, hyy, hyz, hzz;
            compact_hessian_derivatives_from_tile(
                tile, center_index, active, i, j, k,
                ex0, ex1, ex2, kmin_shared, scales,
                hxx, hxy, hxz, hyy, hyz, hzz
            );
            const double laplacian =
                gupxx * hxx + gupyy * hyy + gupzz * hzz +
                2.0 * (gupxy * hxy + gupxz * hxz + gupyz * hyz);
            beta_fields.laplacian[field_index][idx] = laplacian;
            if (field_index == 0) {
                div_hess_x = hxx;
                div_hess_y = hxy;
                div_hess_z = hxz;
            } else if (field_index == 1) {
                div_hess_x += hxy;
                div_hess_y += hyy;
                div_hess_z += hyz;
            } else {
                div_hess_x += hxz;
                div_hess_y += hyz;
                div_hess_z += hzz;
            }
        }
        if (field_index + 1 < COMPACT_BETA_FIELDS) __syncthreads();
    }

    if (valid) {
        beta_fields.divergence[0][idx] = div_hess_x;
        beta_fields.divergence[1][idx] = div_hess_y;
        beta_fields.divergence[2][idx] = div_hess_z;

#pragma unroll 1
        for (int upper = 0; upper < COMPACT_BETA_FIELDS; ++upper) {
            const double contracted =
                gupxx * connection_fields.input[upper][0][idx] +
                gupyy * connection_fields.input[upper][3][idx] +
                gupzz * connection_fields.input[upper][5][idx] +
                2.0 * (
                    gupxy * connection_fields.input[upper][1][idx] +
                    gupxz * connection_fields.input[upper][2][idx] +
                    gupyz * connection_fields.input[upper][4][idx]
                );
            connection_fields.contracted[upper][idx] = contracted;
        }
    }
}

inline void launch_rhs_beta_gamma_prepare_equatorial_compact(
    cudaStream_t stream,
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* gupxx, const double* gupxy, const double* gupxz,
    const double* gupyy, const double* gupyz, const double* gupzz,
    const CompactBetaPrepareFields& beta_fields,
    const CompactConnectionFields& connection_fields
) {
    const dim3 block(
        COMPACT_HESSIAN_BX,
        COMPACT_HESSIAN_BY,
        COMPACT_HESSIAN_BZ
    );
    const dim3 grid(
        (ex0 + block.x - 1) / block.x,
        (ex1 + block.y - 1) / block.y,
        (ex2 + block.z - 1) / block.z
    );
    rhs_beta_gamma_prepare_equatorial_compact_kernel<<<grid, block, 0, stream>>>(
        ex0, ex1, ex2, X, Y, Z,
        gupxx, gupxy, gupxz, gupyy, gupyz, gupzz,
        beta_fields, connection_fields
    );
}

constexpr int COMPACT_TENSOR_COMPONENTS = 6;

struct CompactChiHessianFields {
    const double* gradient[3];
    const double* connection[3][COMPACT_TENSOR_COMPONENTS];
    double* covariant_hessian[COMPACT_TENSOR_COMPONENTS];
};

struct CompactLapseHessianFields {
    const double* connection[3][COMPACT_TENSOR_COMPONENTS];
    const double* inverse_metric[COMPACT_TENSOR_COMPONENTS];
    double* covariant_hessian[COMPACT_TENSOR_COMPONENTS];
    double* trace;
};

__global__ void rhs_source_chi_hessian_equatorial_compact_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* chi,
    CompactChiHessianFields fields
) {
    const int base_i = blockIdx.x * COMPACT_HESSIAN_BX;
    const int base_j = blockIdx.y * COMPACT_HESSIAN_BY;
    const int base_k = blockIdx.z * COMPACT_HESSIAN_BZ;
    const bool block_interior =
        base_i >= COMPACT_HESSIAN_RADIUS &&
        base_j >= COMPACT_HESSIAN_RADIUS &&
        base_k >= COMPACT_HESSIAN_RADIUS &&
        base_i + COMPACT_HESSIAN_BX + COMPACT_HESSIAN_RADIUS <= ex0 &&
        base_j + COMPACT_HESSIAN_BY + COMPACT_HESSIAN_RADIUS <= ex1 &&
        base_k + COMPACT_HESSIAN_BZ + COMPACT_HESSIAN_RADIUS <= ex2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tx + COMPACT_HESSIAN_BX *
        (ty + COMPACT_HESSIAN_BY * tz);
    const int threads = COMPACT_HESSIAN_BX *
        COMPACT_HESSIAN_BY * COMPACT_HESSIAN_BZ;
    const int i = base_i + tx;
    const int j = base_j + ty;
    const int k = base_k + tz;
    const bool valid = i < ex0 && j < ex1 && k < ex2;
    const bool active = valid && i < ex0 - 1 && j < ex1 - 1 && k < ex2 - 1;
    const int idx = valid ? i + ex0 * (j + ex1 * k) : 0;

    __shared__ double tile[COMPACT_HESSIAN_TILE_SIZE];
    __shared__ double scales[12];
    __shared__ int kmin_shared;
    if (tid == 0) {
        const double dx = X[1] - X[0];
        const double dy = Y[1] - Y[0];
        const double dz = Z[1] - Z[0];
        scales[0] = 1.0 / (dx * dx);
        scales[1] = 1.0 / (dy * dy);
        scales[2] = 1.0 / (dz * dz);
        scales[3] = (1.0 / 12.0) / (dx * dx);
        scales[4] = (1.0 / 12.0) / (dy * dy);
        scales[5] = (1.0 / 12.0) / (dz * dz);
        scales[6] = 0.25 / (dx * dy);
        scales[7] = 0.25 / (dx * dz);
        scales[8] = 0.25 / (dy * dz);
        scales[9] = (1.0 / 144.0) / (dx * dy);
        scales[10] = (1.0 / 144.0) / (dx * dz);
        scales[11] = (1.0 / 144.0) / (dy * dz);
        kmin_shared = (fabs(Z[0]) < dz) ? -2 : 0;
    }

    for (int p = tid; p < COMPACT_HESSIAN_LOGICAL_SIZE; p += threads) {
        const int tile_i = p % COMPACT_HESSIAN_SX;
        const int q = p / COMPACT_HESSIAN_SX;
        const int tile_j = q % COMPACT_HESSIAN_SY;
        const int tile_k = q / COMPACT_HESSIAN_SY;
        const int gi = base_i + tile_i - COMPACT_HESSIAN_RADIUS;
        const int gj = base_j + tile_j - COMPACT_HESSIAN_RADIUS;
        const int gk = base_k + tile_k - COMPACT_HESSIAN_RADIUS;
        const int tile_index = compact_hessian_tile_index(
            tile_i, tile_j, tile_k
        );
        tile[tile_index] = block_interior
            ? chi[gi + ex0 * (gj + ex1 * gk)]
            : compact_hessian_symmetry_load(
                chi, gi, gj, gk, ex0, ex1, ex2, 1, 1, 1
            );
    }
    __syncthreads();

    if (valid) {
        double fxx, fxy, fxz, fyy, fyz, fzz;
        compact_hessian_derivatives_from_tile(
            tile,
            compact_hessian_tile_index(
                tx + COMPACT_HESSIAN_RADIUS,
                ty + COMPACT_HESSIAN_RADIUS,
                tz + COMPACT_HESSIAN_RADIUS
            ),
            active, i, j, k, ex0, ex1, ex2, kmin_shared, scales,
            fxx, fxy, fxz, fyy, fyz, fzz
        );
        const double chix = fields.gradient[0][idx];
        const double chiy = fields.gradient[1][idx];
        const double chiz = fields.gradient[2][idx];
        fxx -= fields.connection[0][0][idx] * chix +
               fields.connection[1][0][idx] * chiy +
               fields.connection[2][0][idx] * chiz;
        fxy -= fields.connection[0][1][idx] * chix +
               fields.connection[1][1][idx] * chiy +
               fields.connection[2][1][idx] * chiz;
        fxz -= fields.connection[0][2][idx] * chix +
               fields.connection[1][2][idx] * chiy +
               fields.connection[2][2][idx] * chiz;
        fyy -= fields.connection[0][3][idx] * chix +
               fields.connection[1][3][idx] * chiy +
               fields.connection[2][3][idx] * chiz;
        fyz -= fields.connection[0][4][idx] * chix +
               fields.connection[1][4][idx] * chiy +
               fields.connection[2][4][idx] * chiz;
        fzz -= fields.connection[0][5][idx] * chix +
               fields.connection[1][5][idx] * chiy +
               fields.connection[2][5][idx] * chiz;
        fields.covariant_hessian[0][idx] = fxx;
        fields.covariant_hessian[1][idx] = fxy;
        fields.covariant_hessian[2][idx] = fxz;
        fields.covariant_hessian[3][idx] = fyy;
        fields.covariant_hessian[4][idx] = fyz;
        fields.covariant_hessian[5][idx] = fzz;
    }
}

inline void launch_rhs_source_chi_hessian_equatorial_compact(
    cudaStream_t stream,
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* chi,
    const CompactChiHessianFields& fields
) {
    const dim3 block(
        COMPACT_HESSIAN_BX,
        COMPACT_HESSIAN_BY,
        COMPACT_HESSIAN_BZ
    );
    const dim3 grid(
        (ex0 + block.x - 1) / block.x,
        (ex1 + block.y - 1) / block.y,
        (ex2 + block.z - 1) / block.z
    );
    rhs_source_chi_hessian_equatorial_compact_kernel<<<grid, block, 0, stream>>>(
        ex0, ex1, ex2, X, Y, Z, chi, fields
    );
}

__global__ void rhs_source_lapse_equatorial_compact_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* lapse,
    CompactLapseHessianFields fields
) {
    const int base_i = blockIdx.x * COMPACT_HESSIAN_BX;
    const int base_j = blockIdx.y * COMPACT_HESSIAN_BY;
    const int base_k = blockIdx.z * COMPACT_HESSIAN_BZ;
    const bool block_interior =
        base_i >= COMPACT_HESSIAN_RADIUS &&
        base_j >= COMPACT_HESSIAN_RADIUS &&
        base_k >= COMPACT_HESSIAN_RADIUS &&
        base_i + COMPACT_HESSIAN_BX + COMPACT_HESSIAN_RADIUS <= ex0 &&
        base_j + COMPACT_HESSIAN_BY + COMPACT_HESSIAN_RADIUS <= ex1 &&
        base_k + COMPACT_HESSIAN_BZ + COMPACT_HESSIAN_RADIUS <= ex2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tx + COMPACT_HESSIAN_BX *
        (ty + COMPACT_HESSIAN_BY * tz);
    const int threads = COMPACT_HESSIAN_BX *
        COMPACT_HESSIAN_BY * COMPACT_HESSIAN_BZ;
    const int i = base_i + tx;
    const int j = base_j + ty;
    const int k = base_k + tz;
    const bool valid = i < ex0 && j < ex1 && k < ex2;
    const bool active = valid && i < ex0 - 1 && j < ex1 - 1 && k < ex2 - 1;
    const int idx = valid ? i + ex0 * (j + ex1 * k) : 0;

    __shared__ double tile[COMPACT_HESSIAN_TILE_SIZE];
    __shared__ double scales[18];
    __shared__ int kmin_shared;
    if (tid == 0) {
        const double dx = X[1] - X[0];
        const double dy = Y[1] - Y[0];
        const double dz = Z[1] - Z[0];
        scales[0] = 1.0 / (dx * dx);
        scales[1] = 1.0 / (dy * dy);
        scales[2] = 1.0 / (dz * dz);
        scales[3] = (1.0 / 12.0) / (dx * dx);
        scales[4] = (1.0 / 12.0) / (dy * dy);
        scales[5] = (1.0 / 12.0) / (dz * dz);
        scales[6] = 0.25 / (dx * dy);
        scales[7] = 0.25 / (dx * dz);
        scales[8] = 0.25 / (dy * dz);
        scales[9] = (1.0 / 144.0) / (dx * dy);
        scales[10] = (1.0 / 144.0) / (dx * dz);
        scales[11] = (1.0 / 144.0) / (dy * dz);
        scales[12] = (1.0 / 12.0) / dx;
        scales[13] = (1.0 / 12.0) / dy;
        scales[14] = (1.0 / 12.0) / dz;
        scales[15] = (1.0 / 2.0) / dx;
        scales[16] = (1.0 / 2.0) / dy;
        scales[17] = (1.0 / 2.0) / dz;
        kmin_shared = (fabs(Z[0]) < dz) ? -2 : 0;
    }

    for (int p = tid; p < COMPACT_HESSIAN_LOGICAL_SIZE; p += threads) {
        const int tile_i = p % COMPACT_HESSIAN_SX;
        const int q = p / COMPACT_HESSIAN_SX;
        const int tile_j = q % COMPACT_HESSIAN_SY;
        const int tile_k = q / COMPACT_HESSIAN_SY;
        const int gi = base_i + tile_i - COMPACT_HESSIAN_RADIUS;
        const int gj = base_j + tile_j - COMPACT_HESSIAN_RADIUS;
        const int gk = base_k + tile_k - COMPACT_HESSIAN_RADIUS;
        const int tile_index = compact_hessian_tile_index(
            tile_i, tile_j, tile_k
        );
        tile[tile_index] = block_interior
            ? lapse[gi + ex0 * (gj + ex1 * gk)]
            : compact_hessian_symmetry_load(
                lapse, gi, gj, gk, ex0, ex1, ex2, 1, 1, 1
            );
    }
    __syncthreads();

    if (valid) {
        const int center_index = compact_hessian_tile_index(
            tx + COMPACT_HESSIAN_RADIUS,
            ty + COMPACT_HESSIAN_RADIUS,
            tz + COMPACT_HESSIAN_RADIUS
        );
        double Lapx, Lapy, Lapz;
        compact_first_derivatives_from_tile(
            tile, center_index, active, i, j, k,
            ex0, ex1, ex2, kmin_shared, scales,
            Lapx, Lapy, Lapz
        );
        double fxx, fxy, fxz, fyy, fyz, fzz;
        compact_hessian_derivatives_from_tile(
            tile, center_index, active, i, j, k,
            ex0, ex1, ex2, kmin_shared, scales,
            fxx, fxy, fxz, fyy, fyz, fzz
        );
        fxx -= fields.connection[0][0][idx] * Lapx +
               fields.connection[1][0][idx] * Lapy +
               fields.connection[2][0][idx] * Lapz;
        fyy -= fields.connection[0][3][idx] * Lapx +
               fields.connection[1][3][idx] * Lapy +
               fields.connection[2][3][idx] * Lapz;
        fzz -= fields.connection[0][5][idx] * Lapx +
               fields.connection[1][5][idx] * Lapy +
               fields.connection[2][5][idx] * Lapz;
        fxy -= fields.connection[0][1][idx] * Lapx +
               fields.connection[1][1][idx] * Lapy +
               fields.connection[2][1][idx] * Lapz;
        fxz -= fields.connection[0][2][idx] * Lapx +
               fields.connection[1][2][idx] * Lapy +
               fields.connection[2][2][idx] * Lapz;
        fyz -= fields.connection[0][4][idx] * Lapx +
               fields.connection[1][4][idx] * Lapy +
               fields.connection[2][4][idx] * Lapz;
        const double gupxx = fields.inverse_metric[0][idx];
        const double gupxy = fields.inverse_metric[1][idx];
        const double gupxz = fields.inverse_metric[2][idx];
        const double gupyy = fields.inverse_metric[3][idx];
        const double gupyz = fields.inverse_metric[4][idx];
        const double gupzz = fields.inverse_metric[5][idx];
        const double trace = gupxx * fxx + gupyy * fyy + gupzz * fzz +
                             2.0 * (gupxy * fxy + gupxz * fxz + gupyz * fyz);
        fields.covariant_hessian[0][idx] = fxx;
        fields.covariant_hessian[1][idx] = fxy;
        fields.covariant_hessian[2][idx] = fxz;
        fields.covariant_hessian[3][idx] = fyy;
        fields.covariant_hessian[4][idx] = fyz;
        fields.covariant_hessian[5][idx] = fzz;
        fields.trace[idx] = trace;
    }
}

inline void launch_rhs_source_lapse_equatorial_compact(
    cudaStream_t stream,
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* lapse,
    const CompactLapseHessianFields& fields
) {
    const dim3 block(
        COMPACT_HESSIAN_BX,
        COMPACT_HESSIAN_BY,
        COMPACT_HESSIAN_BZ
    );
    const dim3 grid(
        (ex0 + block.x - 1) / block.x,
        (ex1 + block.y - 1) / block.y,
        (ex2 + block.z - 1) / block.z
    );
    rhs_source_lapse_equatorial_compact_kernel<<<grid, block, 0, stream>>>(
        ex0, ex1, ex2, X, Y, Z, lapse, fields
    );
}

struct CompactChiLapseSourceFields {
    const double* gradient[3];
    double* connection[3][COMPACT_TENSOR_COMPONENTS];
    const double* metric[COMPACT_TENSOR_COMPONENTS];
    const double* inverse_metric[COMPACT_TENSOR_COMPONENTS];
    double* ricci[COMPACT_TENSOR_COMPONENTS];
    double* covariant_hessian[COMPACT_TENSOR_COMPONENTS];
    double* trace;
};

// Chi consumes conformal connections; lapse consumes their physical updates.
__global__ void rhs_source_chi_lapse_equatorial_compact_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* chi, const double* lapse,
    CompactChiLapseSourceFields fields
) {
    const int base_i = blockIdx.x * COMPACT_HESSIAN_BX;
    const int base_j = blockIdx.y * COMPACT_HESSIAN_BY;
    const int base_k = blockIdx.z * COMPACT_HESSIAN_BZ;
    const bool block_interior =
        base_i >= COMPACT_HESSIAN_RADIUS &&
        base_j >= COMPACT_HESSIAN_RADIUS &&
        base_k >= COMPACT_HESSIAN_RADIUS &&
        base_i + COMPACT_HESSIAN_BX + COMPACT_HESSIAN_RADIUS <= ex0 &&
        base_j + COMPACT_HESSIAN_BY + COMPACT_HESSIAN_RADIUS <= ex1 &&
        base_k + COMPACT_HESSIAN_BZ + COMPACT_HESSIAN_RADIUS <= ex2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tx + COMPACT_HESSIAN_BX *
        (ty + COMPACT_HESSIAN_BY * tz);
    const int threads = COMPACT_HESSIAN_BX *
        COMPACT_HESSIAN_BY * COMPACT_HESSIAN_BZ;
    const int i = base_i + tx;
    const int j = base_j + ty;
    const int k = base_k + tz;
    const bool valid = i < ex0 && j < ex1 && k < ex2;
    const bool active = valid && i < ex0 - 1 && j < ex1 - 1 && k < ex2 - 1;
    const int idx = valid ? i + ex0 * (j + ex1 * k) : 0;

    __shared__ double chi_tile[COMPACT_HESSIAN_TILE_SIZE];
    __shared__ double lapse_tile[COMPACT_HESSIAN_TILE_SIZE];
    __shared__ double scales[18];
    __shared__ int kmin_shared;
    if (tid == 0) {
        const double dx = X[1] - X[0];
        const double dy = Y[1] - Y[0];
        const double dz = Z[1] - Z[0];
        scales[0] = 1.0 / (dx * dx);
        scales[1] = 1.0 / (dy * dy);
        scales[2] = 1.0 / (dz * dz);
        scales[3] = (1.0 / 12.0) / (dx * dx);
        scales[4] = (1.0 / 12.0) / (dy * dy);
        scales[5] = (1.0 / 12.0) / (dz * dz);
        scales[6] = 0.25 / (dx * dy);
        scales[7] = 0.25 / (dx * dz);
        scales[8] = 0.25 / (dy * dz);
        scales[9] = (1.0 / 144.0) / (dx * dy);
        scales[10] = (1.0 / 144.0) / (dx * dz);
        scales[11] = (1.0 / 144.0) / (dy * dz);
        scales[12] = (1.0 / 12.0) / dx;
        scales[13] = (1.0 / 12.0) / dy;
        scales[14] = (1.0 / 12.0) / dz;
        scales[15] = (1.0 / 2.0) / dx;
        scales[16] = (1.0 / 2.0) / dy;
        scales[17] = (1.0 / 2.0) / dz;
        kmin_shared = (fabs(Z[0]) < dz) ? -2 : 0;
    }

    for (int p = tid; p < COMPACT_HESSIAN_LOGICAL_SIZE; p += threads) {
        const int tile_i = p % COMPACT_HESSIAN_SX;
        const int q = p / COMPACT_HESSIAN_SX;
        const int tile_j = q % COMPACT_HESSIAN_SY;
        const int tile_k = q / COMPACT_HESSIAN_SY;
        const int gi = base_i + tile_i - COMPACT_HESSIAN_RADIUS;
        const int gj = base_j + tile_j - COMPACT_HESSIAN_RADIUS;
        const int gk = base_k + tile_k - COMPACT_HESSIAN_RADIUS;
        const int tile_index = compact_hessian_tile_index(
            tile_i, tile_j, tile_k
        );
        if (block_interior) {
            const int global_index = gi + ex0 * (gj + ex1 * gk);
            chi_tile[tile_index] = chi[global_index];
            lapse_tile[tile_index] = lapse[global_index];
        } else {
            chi_tile[tile_index] = compact_hessian_symmetry_load(
                chi, gi, gj, gk, ex0, ex1, ex2, 1, 1, 1
            );
            lapse_tile[tile_index] = compact_hessian_symmetry_load(
                lapse, gi, gj, gk, ex0, ex1, ex2, 1, 1, 1
            );
        }
    }
    __syncthreads();

    if (valid) {
        const int center_index = compact_hessian_tile_index(
            tx + COMPACT_HESSIAN_RADIUS,
            ty + COMPACT_HESSIAN_RADIUS,
            tz + COMPACT_HESSIAN_RADIUS
        );

        const double chix = fields.gradient[0][idx];
        const double chiy = fields.gradient[1][idx];
        const double chiz = fields.gradient[2][idx];

        double l_Gamxxx = fields.connection[0][0][idx];
        double l_Gamxxy = fields.connection[0][1][idx];
        double l_Gamxxz = fields.connection[0][2][idx];
        double l_Gamxyy = fields.connection[0][3][idx];
        double l_Gamxyz = fields.connection[0][4][idx];
        double l_Gamxzz = fields.connection[0][5][idx];
        double l_Gamyxx = fields.connection[1][0][idx];
        double l_Gamyxy = fields.connection[1][1][idx];
        double l_Gamyxz = fields.connection[1][2][idx];
        double l_Gamyyy = fields.connection[1][3][idx];
        double l_Gamyyz = fields.connection[1][4][idx];
        double l_Gamyzz = fields.connection[1][5][idx];
        double l_Gamzxx = fields.connection[2][0][idx];
        double l_Gamzxy = fields.connection[2][1][idx];
        double l_Gamzxz = fields.connection[2][2][idx];
        double l_Gamzyy = fields.connection[2][3][idx];
        double l_Gamzyz = fields.connection[2][4][idx];
        double l_Gamzzz = fields.connection[2][5][idx];

        double chi_fxx, chi_fxy, chi_fxz;
        double chi_fyy, chi_fyz, chi_fzz;
        compact_hessian_derivatives_from_tile(
            chi_tile, center_index, active, i, j, k,
            ex0, ex1, ex2, kmin_shared, scales,
            chi_fxx, chi_fxy, chi_fxz,
            chi_fyy, chi_fyz, chi_fzz
        );
        chi_fxx -= l_Gamxxx * chix + l_Gamyxx * chiy + l_Gamzxx * chiz;
        chi_fxy -= l_Gamxxy * chix + l_Gamyxy * chiy + l_Gamzxy * chiz;
        chi_fxz -= l_Gamxxz * chix + l_Gamyxz * chiy + l_Gamzxz * chiz;
        chi_fyy -= l_Gamxyy * chix + l_Gamyyy * chiy + l_Gamzyy * chiz;
        chi_fyz -= l_Gamxyz * chix + l_Gamyyz * chiy + l_Gamzyz * chiz;
        chi_fzz -= l_Gamxzz * chix + l_Gamyzz * chiy + l_Gamzzz * chiz;

        const double chin1 = chi[idx] + 1.0;
        const double l_gxx = fields.metric[0][idx] + 1.0;
        const double l_gxy = fields.metric[1][idx];
        const double l_gxz = fields.metric[2][idx];
        const double l_gyy = fields.metric[3][idx] + 1.0;
        const double l_gyz = fields.metric[4][idx];
        const double l_gzz = fields.metric[5][idx] + 1.0;
        const double gupxx = fields.inverse_metric[0][idx];
        const double gupxy = fields.inverse_metric[1][idx];
        const double gupxz = fields.inverse_metric[2][idx];
        const double gupyy = fields.inverse_metric[3][idx];
        const double gupyz = fields.inverse_metric[4][idx];
        const double gupzz = fields.inverse_metric[5][idx];
        const double f_scalar =
            gupxx * (chi_fxx - 1.5 / chin1 * chix * chix) +
            gupyy * (chi_fyy - 1.5 / chin1 * chiy * chiy) +
            gupzz * (chi_fzz - 1.5 / chin1 * chiz * chiz) +
            2.0 * (gupxy * (chi_fxy - 1.5 / chin1 * chix * chiy) +
                   gupxz * (chi_fxz - 1.5 / chin1 * chix * chiz) +
                   gupyz * (chi_fyz - 1.5 / chin1 * chiy * chiz));
        fields.ricci[0][idx] +=
            (chi_fxx - chix * chix / chin1 / 2.0 + l_gxx * f_scalar) / chin1 / 2.0;
        fields.ricci[3][idx] +=
            (chi_fyy - chiy * chiy / chin1 / 2.0 + l_gyy * f_scalar) / chin1 / 2.0;
        fields.ricci[5][idx] +=
            (chi_fzz - chiz * chiz / chin1 / 2.0 + l_gzz * f_scalar) / chin1 / 2.0;
        fields.ricci[1][idx] +=
            (chi_fxy - chix * chiy / chin1 / 2.0 + l_gxy * f_scalar) / chin1 / 2.0;
        fields.ricci[2][idx] +=
            (chi_fxz - chix * chiz / chin1 / 2.0 + l_gxz * f_scalar) / chin1 / 2.0;
        fields.ricci[4][idx] +=
            (chi_fyz - chiy * chiz / chin1 / 2.0 + l_gyz * f_scalar) / chin1 / 2.0;

        const double gx_phy =
            (gupxx * chix + gupxy * chiy + gupxz * chiz) / chin1;
        const double gy_phy =
            (gupxy * chix + gupyy * chiy + gupyz * chiz) / chin1;
        const double gz_phy =
            (gupxz * chix + gupyz * chiy + gupzz * chiz) / chin1;
        l_Gamxxx -= ((chix + chix) / chin1 - l_gxx * gx_phy) * 0.5;
        l_Gamyxx -= (                         - l_gxx * gy_phy) * 0.5;
        l_Gamzxx -= (                         - l_gxx * gz_phy) * 0.5;
        l_Gamxyy -= (                         - l_gyy * gx_phy) * 0.5;
        l_Gamyyy -= ((chiy + chiy) / chin1 - l_gyy * gy_phy) * 0.5;
        l_Gamzyy -= (                         - l_gyy * gz_phy) * 0.5;
        l_Gamxzz -= (                         - l_gzz * gx_phy) * 0.5;
        l_Gamyzz -= (                         - l_gzz * gy_phy) * 0.5;
        l_Gamzzz -= ((chiz + chiz) / chin1 - l_gzz * gz_phy) * 0.5;
        l_Gamxxy -= (chiy / chin1 - l_gxy * gx_phy) * 0.5;
        l_Gamyxy -= (chix / chin1 - l_gxy * gy_phy) * 0.5;
        l_Gamzxy -= (              - l_gxy * gz_phy) * 0.5;
        l_Gamxxz -= (chiz / chin1 - l_gxz * gx_phy) * 0.5;
        l_Gamyxz -= (              - l_gxz * gy_phy) * 0.5;
        l_Gamzxz -= (chix / chin1 - l_gxz * gz_phy) * 0.5;
        l_Gamxyz -= (              - l_gyz * gx_phy) * 0.5;
        l_Gamyyz -= (chiz / chin1 - l_gyz * gy_phy) * 0.5;
        l_Gamzyz -= (chiy / chin1 - l_gyz * gz_phy) * 0.5;

        fields.connection[0][0][idx] = l_Gamxxx;
        fields.connection[1][0][idx] = l_Gamyxx;
        fields.connection[2][0][idx] = l_Gamzxx;
        fields.connection[0][3][idx] = l_Gamxyy;
        fields.connection[1][3][idx] = l_Gamyyy;
        fields.connection[2][3][idx] = l_Gamzyy;
        fields.connection[0][5][idx] = l_Gamxzz;
        fields.connection[1][5][idx] = l_Gamyzz;
        fields.connection[2][5][idx] = l_Gamzzz;
        fields.connection[0][1][idx] = l_Gamxxy;
        fields.connection[1][1][idx] = l_Gamyxy;
        fields.connection[2][1][idx] = l_Gamzxy;
        fields.connection[0][2][idx] = l_Gamxxz;
        fields.connection[1][2][idx] = l_Gamyxz;
        fields.connection[2][2][idx] = l_Gamzxz;
        fields.connection[0][4][idx] = l_Gamxyz;
        fields.connection[1][4][idx] = l_Gamyyz;
        fields.connection[2][4][idx] = l_Gamzyz;

        double Lapx, Lapy, Lapz;
        compact_first_derivatives_from_tile(
            lapse_tile, center_index, active, i, j, k,
            ex0, ex1, ex2, kmin_shared, scales,
            Lapx, Lapy, Lapz
        );
        double lapse_fxx, lapse_fxy, lapse_fxz;
        double lapse_fyy, lapse_fyz, lapse_fzz;
        compact_hessian_derivatives_from_tile(
            lapse_tile, center_index, active, i, j, k,
            ex0, ex1, ex2, kmin_shared, scales,
            lapse_fxx, lapse_fxy, lapse_fxz,
            lapse_fyy, lapse_fyz, lapse_fzz
        );
        lapse_fxx -= l_Gamxxx * Lapx + l_Gamyxx * Lapy + l_Gamzxx * Lapz;
        lapse_fyy -= l_Gamxyy * Lapx + l_Gamyyy * Lapy + l_Gamzyy * Lapz;
        lapse_fzz -= l_Gamxzz * Lapx + l_Gamyzz * Lapy + l_Gamzzz * Lapz;
        lapse_fxy -= l_Gamxxy * Lapx + l_Gamyxy * Lapy + l_Gamzxy * Lapz;
        lapse_fxz -= l_Gamxxz * Lapx + l_Gamyxz * Lapy + l_Gamzxz * Lapz;
        lapse_fyz -= l_Gamxyz * Lapx + l_Gamyyz * Lapy + l_Gamzyz * Lapz;
        const double trace =
            gupxx * lapse_fxx + gupyy * lapse_fyy + gupzz * lapse_fzz +
            2.0 * (gupxy * lapse_fxy + gupxz * lapse_fxz + gupyz * lapse_fyz);
        fields.covariant_hessian[0][idx] = lapse_fxx;
        fields.covariant_hessian[1][idx] = lapse_fxy;
        fields.covariant_hessian[2][idx] = lapse_fxz;
        fields.covariant_hessian[3][idx] = lapse_fyy;
        fields.covariant_hessian[4][idx] = lapse_fyz;
        fields.covariant_hessian[5][idx] = lapse_fzz;
        fields.trace[idx] = trace;
    }
}

inline void launch_rhs_source_chi_lapse_equatorial_compact(
    cudaStream_t stream,
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* chi, const double* lapse,
    const CompactChiLapseSourceFields& fields
) {
    const dim3 block(
        COMPACT_HESSIAN_BX,
        COMPACT_HESSIAN_BY,
        COMPACT_HESSIAN_BZ
    );
    const dim3 grid(
        (ex0 + block.x - 1) / block.x,
        (ex1 + block.y - 1) / block.y,
        (ex2 + block.z - 1) / block.z
    );
    rhs_source_chi_lapse_equatorial_compact_kernel<<<grid, block, 0, stream>>>(
        ex0, ex1, ex2, X, Y, Z, chi, lapse, fields
    );
}

#endif
