#ifndef ADVECTION_COMPACT_GPU_CUH
#define ADVECTION_COMPACT_GPU_CUH

#include <cuda_runtime.h>
#include <math.h>

constexpr int COMPACT_ADVECTION_FIELDS = 24;
constexpr int COMPACT_ADVECTION_RADIUS = 3;
constexpr int COMPACT_ADVECTION_BX = 8;
constexpr int COMPACT_ADVECTION_BY = 8;
constexpr int COMPACT_ADVECTION_BZ = 4;
constexpr int COMPACT_ADVECTION_SX =
    COMPACT_ADVECTION_BX + 2 * COMPACT_ADVECTION_RADIUS;
constexpr int COMPACT_ADVECTION_SY =
    COMPACT_ADVECTION_BY + 2 * COMPACT_ADVECTION_RADIUS;
constexpr int COMPACT_ADVECTION_SZ =
    COMPACT_ADVECTION_BZ + 2 * COMPACT_ADVECTION_RADIUS;
constexpr int COMPACT_ADVECTION_TILE_SIZE =
    COMPACT_ADVECTION_SX * COMPACT_ADVECTION_SY * COMPACT_ADVECTION_SZ;

struct CompactAdvectionFields {
    const double* input[COMPACT_ADVECTION_FIELDS];
    double* rhs[COMPACT_ADVECTION_FIELDS];
    int parity_x[COMPACT_ADVECTION_FIELDS];
    int parity_y[COMPACT_ADVECTION_FIELDS];
    int parity_z[COMPACT_ADVECTION_FIELDS];
};

__device__ __forceinline__ int compact_tile_index(int x, int y, int z) {
    return x + COMPACT_ADVECTION_SX * (y + COMPACT_ADVECTION_SY * z);
}

// The legacy symmetry_bd helper stores three lower ghost layers at indices
// -3..-1.  Keep the same reflection and sign convention in the tile loader.
__device__ __forceinline__ double compact_symmetry_load(
    const double* field,
    int i, int j, int k,
    int ex0, int ex1, int ex2,
    int parity_x, int parity_y, int parity_z
) {
    if (i < -COMPACT_ADVECTION_RADIUS || i >= ex0 ||
        j < -COMPACT_ADVECTION_RADIUS || j >= ex1 ||
        k < -COMPACT_ADVECTION_RADIUS || k >= ex2) {
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

__device__ __forceinline__ double compact_tile_at(
    const double* tile, int center_index, int di, int dj, int dk
) {
    return tile[center_index + di + COMPACT_ADVECTION_SX *
        (dj + COMPACT_ADVECTION_SY * dk)];
}

template <bool boundary>
__device__ __forceinline__ double compact_lopsided(
    const double* tile,
    int center_index,
    int i, int j, int k,
    int imin, int jmin, int kmin,
    int imax, int jmax, int kmax,
    double vx, double vy, double vz,
    double d12dx, double d12dy, double d12dz
) {
    const double center = compact_tile_at(tile, center_index, 0, 0, 0);
    double value = 0.0;

#define COMPACT_AT(di, dj, dk) compact_tile_at(tile, center_index, di, dj, dk)
#define COMPACT_FORWARD_X \
    (-3.0 * COMPACT_AT(-1, 0, 0) - 10.0 * center + \
     18.0 * COMPACT_AT(1, 0, 0) - 6.0 * COMPACT_AT(2, 0, 0) + \
     COMPACT_AT(3, 0, 0))
#define COMPACT_BACKWARD_X \
    (-3.0 * COMPACT_AT(1, 0, 0) - 10.0 * center + \
     18.0 * COMPACT_AT(-1, 0, 0) - 6.0 * COMPACT_AT(-2, 0, 0) + \
     COMPACT_AT(-3, 0, 0))
#define COMPACT_CENTERED_X \
    (COMPACT_AT(-2, 0, 0) - 8.0 * COMPACT_AT(-1, 0, 0) + \
     8.0 * COMPACT_AT(1, 0, 0) - COMPACT_AT(2, 0, 0))

    if (!boundary) {
        if (vx > 0.0) value += vx * d12dx * COMPACT_FORWARD_X;
        else if (vx < 0.0) value -= vx * d12dx * COMPACT_BACKWARD_X;
    } else if (vx > 0.0) {
        if (i + 3 <= imax) value += vx * d12dx * COMPACT_FORWARD_X;
        else if (i + 2 <= imax) value += vx * d12dx * COMPACT_CENTERED_X;
        else if (i + 1 <= imax) value -= vx * d12dx * COMPACT_BACKWARD_X;
    } else if (vx < 0.0) {
        if (i - 3 >= imin) value -= vx * d12dx * COMPACT_BACKWARD_X;
        else if (i - 2 >= imin) value += vx * d12dx * COMPACT_CENTERED_X;
        else if (i - 1 >= imin) value += vx * d12dx * COMPACT_FORWARD_X;
    }

#undef COMPACT_FORWARD_X
#undef COMPACT_BACKWARD_X
#undef COMPACT_CENTERED_X
#define COMPACT_FORWARD_Y \
    (-3.0 * COMPACT_AT(0, -1, 0) - 10.0 * center + \
     18.0 * COMPACT_AT(0, 1, 0) - 6.0 * COMPACT_AT(0, 2, 0) + \
     COMPACT_AT(0, 3, 0))
#define COMPACT_BACKWARD_Y \
    (-3.0 * COMPACT_AT(0, 1, 0) - 10.0 * center + \
     18.0 * COMPACT_AT(0, -1, 0) - 6.0 * COMPACT_AT(0, -2, 0) + \
     COMPACT_AT(0, -3, 0))
#define COMPACT_CENTERED_Y \
    (COMPACT_AT(0, -2, 0) - 8.0 * COMPACT_AT(0, -1, 0) + \
     8.0 * COMPACT_AT(0, 1, 0) - COMPACT_AT(0, 2, 0))

    if (!boundary) {
        if (vy > 0.0) value += vy * d12dy * COMPACT_FORWARD_Y;
        else if (vy < 0.0) value -= vy * d12dy * COMPACT_BACKWARD_Y;
    } else if (vy > 0.0) {
        if (j + 3 <= jmax) value += vy * d12dy * COMPACT_FORWARD_Y;
        else if (j + 2 <= jmax) value += vy * d12dy * COMPACT_CENTERED_Y;
        else if (j + 1 <= jmax) value -= vy * d12dy * COMPACT_BACKWARD_Y;
    } else if (vy < 0.0) {
        if (j - 3 >= jmin) value -= vy * d12dy * COMPACT_BACKWARD_Y;
        else if (j - 2 >= jmin) value += vy * d12dy * COMPACT_CENTERED_Y;
        else if (j - 1 >= jmin) value += vy * d12dy * COMPACT_FORWARD_Y;
    }

#undef COMPACT_FORWARD_Y
#undef COMPACT_BACKWARD_Y
#undef COMPACT_CENTERED_Y
#define COMPACT_FORWARD_Z \
    (-3.0 * COMPACT_AT(0, 0, -1) - 10.0 * center + \
     18.0 * COMPACT_AT(0, 0, 1) - 6.0 * COMPACT_AT(0, 0, 2) + \
     COMPACT_AT(0, 0, 3))
#define COMPACT_BACKWARD_Z \
    (-3.0 * COMPACT_AT(0, 0, 1) - 10.0 * center + \
     18.0 * COMPACT_AT(0, 0, -1) - 6.0 * COMPACT_AT(0, 0, -2) + \
     COMPACT_AT(0, 0, -3))
#define COMPACT_CENTERED_Z \
    (COMPACT_AT(0, 0, -2) - 8.0 * COMPACT_AT(0, 0, -1) + \
     8.0 * COMPACT_AT(0, 0, 1) - COMPACT_AT(0, 0, 2))

    if (!boundary) {
        if (vz > 0.0) value += vz * d12dz * COMPACT_FORWARD_Z;
        else if (vz < 0.0) value -= vz * d12dz * COMPACT_BACKWARD_Z;
    } else if (vz > 0.0) {
        if (k + 3 <= kmax) value += vz * d12dz * COMPACT_FORWARD_Z;
        else if (k + 2 <= kmax) value += vz * d12dz * COMPACT_CENTERED_Z;
        else if (k + 1 <= kmax) value -= vz * d12dz * COMPACT_BACKWARD_Z;
    } else if (vz < 0.0) {
        if (k - 3 >= kmin) value -= vz * d12dz * COMPACT_BACKWARD_Z;
        else if (k - 2 >= kmin) value += vz * d12dz * COMPACT_CENTERED_Z;
        else if (k - 1 >= kmin) value += vz * d12dz * COMPACT_FORWARD_Z;
    }

#undef COMPACT_FORWARD_Z
#undef COMPACT_BACKWARD_Z
#undef COMPACT_CENTERED_Z
#undef COMPACT_AT
    return value;
}

__device__ __forceinline__ double compact_ko(
    const double* tile, int center_index,
    double dx, double dy, double dz, double eps
) {
#define COMPACT_KO_AT(di, dj, dk) compact_tile_at(tile, center_index, di, dj, dk)
    const double center = COMPACT_KO_AT(0, 0, 0);
    const double x = (COMPACT_KO_AT(-3, 0, 0) + COMPACT_KO_AT(3, 0, 0))
        - 6.0 * (COMPACT_KO_AT(-2, 0, 0) + COMPACT_KO_AT(2, 0, 0))
        + 15.0 * (COMPACT_KO_AT(-1, 0, 0) + COMPACT_KO_AT(1, 0, 0))
        - 20.0 * center;
    const double y = (COMPACT_KO_AT(0, -3, 0) + COMPACT_KO_AT(0, 3, 0))
        - 6.0 * (COMPACT_KO_AT(0, -2, 0) + COMPACT_KO_AT(0, 2, 0))
        + 15.0 * (COMPACT_KO_AT(0, -1, 0) + COMPACT_KO_AT(0, 1, 0))
        - 20.0 * center;
    const double z = (COMPACT_KO_AT(0, 0, -3) + COMPACT_KO_AT(0, 0, 3))
        - 6.0 * (COMPACT_KO_AT(0, 0, -2) + COMPACT_KO_AT(0, 0, 2))
        + 15.0 * (COMPACT_KO_AT(0, 0, -1) + COMPACT_KO_AT(0, 0, 1))
        - 20.0 * center;
#undef COMPACT_KO_AT
    return eps / 64.0 * (x / dx + y / dy + z / dz);
}

// A block reuses one full radius-three tile across all 24 fields.  Linear
// cooperative loads keep work balanced and coalesced; the precomputed center
// index turns every stencil access into a constant shared-memory offset.
__global__ void rhs_advection_equatorial_compact_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* betax, const double* betay, const double* betaz,
    CompactAdvectionFields fields, double eps,
    double gauge_ff, double gauge_eta
) {
    const int base_i = blockIdx.x * COMPACT_ADVECTION_BX;
    const int base_j = blockIdx.y * COMPACT_ADVECTION_BY;
    const int base_k = blockIdx.z * COMPACT_ADVECTION_BZ;
    const bool block_interior =
        base_i >= COMPACT_ADVECTION_RADIUS &&
        base_j >= COMPACT_ADVECTION_RADIUS &&
        base_k >= COMPACT_ADVECTION_RADIUS &&
        base_i + COMPACT_ADVECTION_BX + COMPACT_ADVECTION_RADIUS <= ex0 &&
        base_j + COMPACT_ADVECTION_BY + COMPACT_ADVECTION_RADIUS <= ex1 &&
        base_k + COMPACT_ADVECTION_BZ + COMPACT_ADVECTION_RADIUS <= ex2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tx + COMPACT_ADVECTION_BX *
        (ty + COMPACT_ADVECTION_BY * tz);
    const int threads = COMPACT_ADVECTION_BX *
        COMPACT_ADVECTION_BY * COMPACT_ADVECTION_BZ;
    const int i = base_i + tx;
    const int j = base_j + ty;
    const int k = base_k + tz;
    const bool valid = i < ex0 && j < ex1 && k < ex2;
    const bool active = valid && i < ex0 - 1 && j < ex1 - 1 && k < ex2 - 1;
    const int idx = valid ? i + ex0 * (j + ex1 * k) : 0;

    __shared__ double tile[COMPACT_ADVECTION_TILE_SIZE];
    __shared__ double gamma_source[3][
        COMPACT_ADVECTION_BX * COMPACT_ADVECTION_BY * COMPACT_ADVECTION_BZ];
    __shared__ double scales[6];
    __shared__ int kmin_shared;
    if (tid == 0) {
        const double dx = X[1] - X[0];
        const double dy = Y[1] - Y[0];
        const double dz = Z[1] - Z[0];
        scales[0] = 1.0 / (12.0 * dx);
        scales[1] = 1.0 / (12.0 * dy);
        scales[2] = 1.0 / (12.0 * dz);
        scales[3] = dx;
        scales[4] = dy;
        scales[5] = dz;
        kmin_shared = (fabs(Z[0]) < dz) ? -3 : 0;
    }
    __syncthreads();

    const double vx = valid ? betax[idx] : 0.0;
    const double vy = valid ? betay[idx] : 0.0;
    const double vz = valid ? betaz[idx] : 0.0;
    const int kmin = kmin_shared;

    if (active) {
        gamma_source[0][tid] = fields.rhs[14][idx];
        gamma_source[1][tid] = fields.rhs[15][idx];
        gamma_source[2][tid] = fields.rhs[16][idx];
    } else if (valid) {
        fields.rhs[17][idx] =
            -2.0 * (fields.input[17][idx] + 1.0) * fields.input[13][idx];
        fields.rhs[18][idx] = gauge_ff * fields.input[21][idx];
        fields.rhs[19][idx] = gauge_ff * fields.input[22][idx];
        fields.rhs[20][idx] = gauge_ff * fields.input[23][idx];
        fields.rhs[21][idx] =
            fields.rhs[14][idx] - gauge_eta * fields.input[21][idx];
        fields.rhs[22][idx] =
            fields.rhs[15][idx] - gauge_eta * fields.input[22][idx];
        fields.rhs[23][idx] =
            fields.rhs[16][idx] - gauge_eta * fields.input[23][idx];
    }

    const int center_index = compact_tile_index(
        tx + COMPACT_ADVECTION_RADIUS,
        ty + COMPACT_ADVECTION_RADIUS,
        tz + COMPACT_ADVECTION_RADIUS
    );

#pragma unroll 1
    for (int field_index = 0; field_index < COMPACT_ADVECTION_FIELDS; ++field_index) {
        const double* field = fields.input[field_index];
        const int parity_x = fields.parity_x[field_index];
        const int parity_y = fields.parity_y[field_index];
        const int parity_z = fields.parity_z[field_index];
        for (int p = tid; p < COMPACT_ADVECTION_TILE_SIZE; p += threads) {
            const int tile_i = p % COMPACT_ADVECTION_SX;
            const int q = p / COMPACT_ADVECTION_SX;
            const int tile_j = q % COMPACT_ADVECTION_SY;
            const int tile_k = q / COMPACT_ADVECTION_SY;
            const int gi = base_i + tile_i - COMPACT_ADVECTION_RADIUS;
            const int gj = base_j + tile_j - COMPACT_ADVECTION_RADIUS;
            const int gk = base_k + tile_k - COMPACT_ADVECTION_RADIUS;
            tile[p] = block_interior
                ? field[gi + ex0 * (gj + ex1 * gk)]
                : compact_symmetry_load(
                    field, gi, gj, gk, ex0, ex1, ex2,
                    parity_x, parity_y, parity_z
                );
        }
        __syncthreads();

        if (active) {
            const double advection = block_interior
                ? compact_lopsided<false>(
                    tile, center_index, i, j, k,
                    0, 0, kmin, ex0 - 1, ex1 - 1, ex2 - 1,
                    vx, vy, vz, scales[0], scales[1], scales[2]
                )
                : compact_lopsided<true>(
                    tile, center_index, i, j, k,
                    0, 0, kmin, ex0 - 1, ex1 - 1, ex2 - 1,
                    vx, vy, vz, scales[0], scales[1], scales[2]
                );
            const double center = compact_tile_at(
                tile, center_index, 0, 0, 0
            );
            double base;
            if (field_index == 17) {
                base = -2.0 * (center + 1.0) * fields.input[13][idx];
            } else if (field_index >= 18 && field_index <= 20) {
                base = gauge_ff * fields.input[field_index + 3][idx];
            } else if (field_index >= 21) {
                base = gamma_source[field_index - 21][tid] -
                    gauge_eta * center;
            } else {
                base = fields.rhs[field_index][idx];
            }
            double value = base + advection;
            if (eps > 0.0 &&
                i >= 3 && i + 3 <= ex0 - 1 &&
                j >= 3 && j + 3 <= ex1 - 1 &&
                k - 3 >= kmin && k + 3 <= ex2 - 1) {
                value += compact_ko(
                    tile, center_index,
                    scales[3], scales[4], scales[5], eps
                );
            }
            fields.rhs[field_index][idx] = value;
        }
        __syncthreads();
    }
}

inline void launch_rhs_advection_equatorial_compact(
    cudaStream_t stream,
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    const double* betax, const double* betay, const double* betaz,
    const CompactAdvectionFields& fields, double eps,
    double gauge_ff, double gauge_eta
) {
    const dim3 block(
        COMPACT_ADVECTION_BX,
        COMPACT_ADVECTION_BY,
        COMPACT_ADVECTION_BZ
    );
    const dim3 grid(
        (ex0 + block.x - 1) / block.x,
        (ex1 + block.y - 1) / block.y,
        (ex2 + block.z - 1) / block.z
    );
    rhs_advection_equatorial_compact_kernel<<<grid, block, 0, stream>>>(
        ex0, ex1, ex2, X, Y, Z, betax, betay, betaz, fields, eps,
        gauge_ff, gauge_eta
    );
}

#endif
