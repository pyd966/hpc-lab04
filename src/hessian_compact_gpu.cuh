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

#endif
