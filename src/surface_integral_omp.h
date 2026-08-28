#ifndef SURFACE_INTEGRAL_OMP_H
#define SURFACE_INTEGRAL_OMP_H

#include <algorithm>
#include <vector>
#include <omp.h>

namespace {

const int omp_interp_order = 2 * ghost_width;

struct OmpCachedPoint
{
    Block *block;
    int index[3][2 * ghost_width];
    unsigned char reflected[3][2 * ghost_width];
    double coefficient[3][2 * ghost_width];
};

struct OmpSpherePlan
{
    Patch *patch;
    int level;
    double radius;
    std::vector<Block *> blocks;
    std::vector<OmpCachedPoint> points;
};

struct OmpWaveBasis
{
    bool ready;
    int symmetry;
    int spinw;
    int maxl;
    int modes;
    int points;
    double real_symmetry[3];
    double imag_symmetry[3];
    std::vector<double> real_from_real;
    std::vector<double> real_from_imag;
    std::vector<double> imag_from_real;
    std::vector<double> imag_from_imag;

    OmpWaveBasis() : ready(false), symmetry(-1), spinw(0), maxl(0),
                     modes(0), points(0)
    {
        std::fill(real_symmetry, real_symmetry + 3, 0.0);
        std::fill(imag_symmetry, imag_symmetry + 3, 0.0);
    }
};

struct OmpAnalysisCache
{
    std::vector<OmpSpherePlan> sphere_plans;
    OmpWaveBasis wave_basis;
    std::vector<double> wave_partials;
    std::vector<double> adm_partials;
};

static std::vector<Block *> omp_collect_blocks(Patch *patch)
{
    std::vector<Block *> blocks;
    MyList<Block> *node = patch->blb;
    while (node)
    {
        blocks.push_back(node->data);
        if (node == patch->ble)
            break;
        node = node->next;
    }
    return blocks;
}

static void omp_lagrange_coefficients(double target, double *coefficient)
{
    for (int i = 0; i < omp_interp_order; ++i)
    {
        double value = 1.0;
        for (int j = 0; j < omp_interp_order; ++j)
            if (j != i)
                value *= (target - j) / static_cast<double>(i - j);
        coefficient[i] = value;
    }
}

static OmpCachedPoint omp_build_cached_point(
    Patch *patch, const std::vector<Block *> &blocks,
    double x, double y, double z, int symmetry)
{
    const double coordinate[3] = {x, y, z};
    double spacing[3];
    for (int direction = 0; direction < 3; ++direction)
    {
        spacing[direction] = patch->getdX(direction);
        const double lower = patch->bbox[direction] +
                             patch->lli[direction] * spacing[direction];
        const double upper = patch->bbox[3 + direction] -
                             patch->uui[direction] * spacing[direction];
        if (coordinate[direction] < lower || coordinate[direction] > upper)
        {
            std::cerr << "surface_integral: analysis point outside Patch" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    Block *owner = 0;
    for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index)
    {
        Block *candidate = blocks[block_index];
        bool contains = true;
        for (int direction = 0; direction < 3; ++direction)
        {
            const double lower =
                feq(candidate->bbox[direction], patch->bbox[direction],
                    spacing[direction] / 2.0)
                    ? candidate->bbox[direction] +
                          patch->lli[direction] * spacing[direction]
                    : candidate->bbox[direction] +
                          ghost_width * spacing[direction];
            const double upper =
                feq(candidate->bbox[3 + direction], patch->bbox[3 + direction],
                    spacing[direction] / 2.0)
                    ? candidate->bbox[3 + direction] -
                          patch->uui[direction] * spacing[direction]
                    : candidate->bbox[3 + direction] -
                          ghost_width * spacing[direction];
            if (coordinate[direction] - lower < -spacing[direction] / 2.0 ||
                coordinate[direction] - upper > spacing[direction] / 2.0)
            {
                contains = false;
                break;
            }
        }
        if (contains)
        {
            owner = candidate;
            break;
        }
    }

    if (!owner)
    {
        std::cerr << "surface_integral: no Block owns analysis point" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    OmpCachedPoint result;
    result.block = owner;
    for (int direction = 0; direction < 3; ++direction)
    {
        const double dx = owner->X[direction][1] - owner->X[direction][0];
        const int center =
            static_cast<int>((coordinate[direction] -
                              owner->X[direction][0]) /
                                 dx +
                             0.4) +
            1;
        int base = center - omp_interp_order / 2 + 1;
        int minimum = 1;
        if (symmetry == 2 && direction < 2 &&
            fabs(owner->X[direction][0]) < dx)
            minimum = -omp_interp_order / 2 + 1;
        if (symmetry != 0 && direction == 2 &&
            fabs(owner->X[direction][0]) < dx)
            minimum = -omp_interp_order / 2 + 1;

        if (base < minimum)
            base = minimum;
        if (base + omp_interp_order - 1 > owner->shape[direction])
            base = owner->shape[direction] + 1 - omp_interp_order;

        const double local_coordinate =
            base > 0
                ? (coordinate[direction] -
                   owner->X[direction][base - 1]) /
                      dx
                : (coordinate[direction] +
                   owner->X[direction][-base]) /
                      dx;
        omp_lagrange_coefficients(local_coordinate,
                                  result.coefficient[direction]);

        for (int offset = 0; offset < omp_interp_order; ++offset)
        {
            const int fortran_index = base + offset;
            result.reflected[direction][offset] =
                fortran_index <= 0 ? 1 : 0;
            result.index[direction][offset] =
                fortran_index <= 0 ? -fortran_index : fortran_index - 1;
        }
    }
    return result;
}

static OmpSpherePlan &omp_get_sphere_plan(
    OmpAnalysisCache *cache, Patch *patch, int level, double radius,
    int points, const double *nx, const double *ny, const double *nz,
    int symmetry)
{
    const std::vector<Block *> blocks = omp_collect_blocks(patch);
    for (std::size_t i = 0; i < cache->sphere_plans.size(); ++i)
    {
        OmpSpherePlan &candidate = cache->sphere_plans[i];
        if (candidate.patch == patch && candidate.level == level &&
            candidate.radius == radius && candidate.blocks == blocks)
            return candidate;
    }

    OmpSpherePlan plan;
    plan.patch = patch;
    plan.level = level;
    plan.radius = radius;
    plan.blocks = blocks;
    plan.points.resize(points);
    for (int point = 0; point < points; ++point)
        plan.points[point] = omp_build_cached_point(
            patch, blocks, radius * nx[point], radius * ny[point],
            radius * nz[point], symmetry);
    cache->sphere_plans.push_back(plan);
    return cache->sphere_plans.back();
}

static double omp_cached_interpolate(const OmpCachedPoint &point,
                                     const var *field)
{
    const Block *block = point.block;
    const double *data = block->fgfs[field->sgfn];
    double x_values[omp_interp_order];

    for (int x = 0; x < omp_interp_order; ++x)
    {
        double y_values[omp_interp_order];
        const double x_sign =
            point.reflected[0][x] ? field->SoA[0] : 1.0;
        for (int y = 0; y < omp_interp_order; ++y)
        {
            double z_value = 0.0;
            const double xy_sign =
                x_sign * (point.reflected[1][y] ? field->SoA[1] : 1.0);
            for (int z = 0; z < omp_interp_order; ++z)
            {
                const int offset =
                    point.index[0][x] +
                    block->shape[0] *
                        (point.index[1][y] +
                         block->shape[1] * point.index[2][z]);
                const double sign =
                    xy_sign *
                    (point.reflected[2][z] ? field->SoA[2] : 1.0);
                z_value += point.coefficient[2][z] * sign * data[offset];
            }
            y_values[y] = z_value;
        }

        double y_value = 0.0;
        for (int y = 0; y < omp_interp_order; ++y)
            y_value += point.coefficient[1][y] * y_values[y];
        x_values[x] = y_value;
    }

    double result = 0.0;
    for (int x = 0; x < omp_interp_order; ++x)
        result += point.coefficient[0][x] * x_values[x];
    return result;
}

static bool omp_wave_basis_matches(
    const OmpWaveBasis &basis, int symmetry, int spinw, int maxl,
    int modes, int points, const var *real_field, const var *imag_field)
{
    if (!basis.ready || basis.symmetry != symmetry ||
        basis.spinw != spinw || basis.maxl != maxl ||
        basis.modes != modes || basis.points != points)
        return false;
    for (int direction = 0; direction < 3; ++direction)
        if (basis.real_symmetry[direction] != real_field->SoA[direction] ||
            basis.imag_symmetry[direction] != imag_field->SoA[direction])
            return false;
    return true;
}

static OmpWaveBasis &omp_get_wave_basis(
    OmpAnalysisCache *cache, int symmetry, int spinw, int maxl, int modes,
    int points, int n_phi, double dphi, const double *costheta,
    const double *theta_weight, const var *real_field, const var *imag_field)
{
    OmpWaveBasis &basis = cache->wave_basis;
    if (omp_wave_basis_matches(basis, symmetry, spinw, maxl, modes,
                               points, real_field, imag_field))
        return basis;

    basis.ready = true;
    basis.symmetry = symmetry;
    basis.spinw = spinw;
    basis.maxl = maxl;
    basis.modes = modes;
    basis.points = points;
    for (int direction = 0; direction < 3; ++direction)
    {
        basis.real_symmetry[direction] = real_field->SoA[direction];
        basis.imag_symmetry[direction] = imag_field->SoA[direction];
    }

    const std::size_t count =
        static_cast<std::size_t>(points) * static_cast<std::size_t>(modes);
    basis.real_from_real.assign(count, 0.0);
    basis.real_from_imag.assign(count, 0.0);
    basis.imag_from_real.assign(count, 0.0);
    basis.imag_from_imag.assign(count, 0.0);

    const int replicas = symmetry == 0 ? 1 : (symmetry == 1 ? 2 : 8);
    for (int point = 0; point < points; ++point)
    {
        const int theta_index = point / n_phi;
        const int phi_index = point - theta_index * n_phi;
        const double phi = (phi_index + 0.5) * dphi;
        int mode = 0;
        for (int l = spinw; l <= maxl; ++l)
            for (int m = -l; m <= l; ++m, ++mode)
            {
                const std::size_t index =
                    static_cast<std::size_t>(point) * modes + mode;
                for (int replica = 0; replica < replicas; ++replica)
                {
                    double reflected_costheta =
                        (replica == 1 || replica == 3 ||
                         replica == 5 || replica == 7)
                            ? -costheta[theta_index]
                            : costheta[theta_index];
                    double reflected_phi = phi;
                    if (replica == 2 || replica == 3)
                        reflected_phi = -phi;
                    else if (replica == 4 || replica == 5)
                        reflected_phi = M_PI - phi;
                    else if (replica == 6 || replica == 7)
                        reflected_phi = M_PI + phi;

                    double real_sign = 1.0;
                    double imag_sign = 1.0;
                    if (replica & 1)
                    {
                        real_sign *= real_field->SoA[2];
                        imag_sign *= imag_field->SoA[2];
                    }
                    if (replica == 2 || replica == 3 ||
                        replica == 6 || replica == 7)
                    {
                        real_sign *= real_field->SoA[1];
                        imag_sign *= imag_field->SoA[1];
                    }
                    if (replica >= 4)
                    {
                        real_sign *= real_field->SoA[0];
                        imag_sign *= imag_field->SoA[0];
                    }

                    const double theta =
                        sqrt((2 * l + 1.0) / (4.0 * M_PI)) *
                        misc::Wigner_d_function(l, m, spinw,
                                                reflected_costheta) *
                        theta_weight[theta_index];
                    const double cosine = cos(m * reflected_phi);
                    const double sine = sin(m * reflected_phi);
                    basis.real_from_real[index] +=
                        theta * real_sign * cosine;
                    basis.real_from_imag[index] +=
                        theta * imag_sign * sine;
                    basis.imag_from_real[index] -=
                        theta * real_sign * sine;
                    basis.imag_from_imag[index] +=
                        theta * imag_sign * cosine;
                }
            }
    }
    return basis;
}

static void omp_surface_wave(
    OmpAnalysisCache *cache, double radius, int level, cgh *gh,
    var *real_field, var *imag_field, int spinw, int maxl, int modes,
    double *real_output, double *imag_output, int symmetry, int points,
    int n_phi, double dphi, const double *costheta,
    const double *theta_weight, const double *nx, const double *ny,
    const double *nz)
{
    OmpSpherePlan &plan = omp_get_sphere_plan(
        cache, gh->PatL[level]->data, level, radius, points,
        nx, ny, nz, symmetry);
    OmpWaveBasis &basis = omp_get_wave_basis(
        cache, symmetry, spinw, maxl, modes, points, n_phi, dphi,
        costheta, theta_weight, real_field, imag_field);

    const int workers = omp_get_max_threads();
    cache->wave_partials.assign(
        static_cast<std::size_t>(workers) * 2 * modes, 0.0);

#pragma omp parallel
    {
        const int worker = omp_get_thread_num();
        double *local =
            &cache->wave_partials[static_cast<std::size_t>(worker) *
                                  2 * modes];
#pragma omp for schedule(static)
        for (int point = 0; point < points; ++point)
        {
            const double real_value =
                omp_cached_interpolate(plan.points[point], real_field);
            const double imag_value =
                omp_cached_interpolate(plan.points[point], imag_field);
            const std::size_t base =
                static_cast<std::size_t>(point) * modes;
            for (int mode = 0; mode < modes; ++mode)
            {
                const std::size_t index = base + mode;
                local[mode] +=
                    basis.real_from_real[index] * real_value +
                    basis.real_from_imag[index] * imag_value;
                local[modes + mode] +=
                    basis.imag_from_real[index] * real_value +
                    basis.imag_from_imag[index] * imag_value;
            }
        }
    }

    for (int mode = 0; mode < modes; ++mode)
    {
        double real_sum = 0.0;
        double imag_sum = 0.0;
        for (int worker = 0; worker < workers; ++worker)
        {
            const std::size_t base =
                static_cast<std::size_t>(worker) * 2 * modes;
            real_sum += cache->wave_partials[base + mode];
            imag_sum += cache->wave_partials[base + modes + mode];
        }
        real_output[mode] = real_sum * radius * dphi;
        imag_output[mode] = imag_sum * radius * dphi;
    }
}

static void omp_accumulate_adm_point(
    const double *value, double x, double y, double z,
    double nx, double ny, double nz, double weight,
    int symmetry, double *output)
{
    double chi = value[3];
    const double trk = value[4];
    const double gxx = value[5] + 1.0;
    const double gxy = value[6];
    const double gxz = value[7];
    const double gyy = value[8] + 1.0;
    const double gyz = value[9];
    const double gzz = value[10] + 1.0;
    double axx = value[11];
    double axy = value[12];
    double axz = value[13];
    double ayy = value[14];
    double ayz = value[15];
    double azz = value[16];

    chi = 1.0 / (1.0 + chi);
    const double psi = chi * sqrt(chi);
    output[0] +=
        (value[0] * nx + value[1] * ny + value[2] * nz) * weight;

    double determinant =
        gxx * gyy * gzz + gxy * gyz * gxz + gxz * gxy * gyz -
        gxz * gyy * gxz - gxy * gxy * gzz - gxx * gyz * gyz;
    const double gupxx = (gyy * gzz - gyz * gyz) / determinant;
    const double gupxy = -(gxy * gzz - gyz * gxz) / determinant;
    const double gupxz = (gxy * gyz - gyy * gxz) / determinant;
    const double gupyy = (gxx * gzz - gxz * gxz) / determinant;
    const double gupyz = -(gxx * gyz - gxy * gxz) / determinant;
    const double gupzz = (gxx * gyy - gxy * gxy) / determinant;

    const double aupxx = gupxx * axx + gupxy * axy + gupxz * axz;
    const double aupxy = gupxx * axy + gupxy * ayy + gupxz * ayz;
    const double aupxz = gupxx * axz + gupxy * ayz + gupxz * azz;
    const double aupyx = gupxy * axx + gupyy * axy + gupyz * axz;
    const double aupyy = gupxy * axy + gupyy * ayy + gupyz * ayz;
    const double aupyz = gupxy * axz + gupyy * ayz + gupyz * azz;
    const double aupzx = gupxz * axx + gupyz * axy + gupzz * axz;
    const double aupzy = gupxz * axy + gupyz * ayy + gupzz * ayz;
    const double aupzz = gupxz * axz + gupyz * ayz + gupzz * azz;
    const double one_eighth = 0.125;

    if (symmetry == 0)
    {
        output[4] += one_eighth * psi *
            (nx * (y * aupxz - z * aupxy) +
             ny * (y * aupyz - z * aupyy) +
             nz * (y * aupzz - z * aupzy)) * weight;
        output[5] += one_eighth * psi *
            (nx * (z * aupxx - x * aupxz) +
             ny * (z * aupyx - x * aupyz) +
             nz * (z * aupzx - x * aupzz)) * weight;
        output[6] += one_eighth * psi *
            (nx * (x * aupxy - y * aupxx) +
             ny * (x * aupyy - y * aupyx) +
             nz * (x * aupzy - y * aupzx)) * weight;
    }
    else if (symmetry == 1)
    {
        output[6] += one_eighth * psi *
            (nx * (x * aupxy - y * aupxx) +
             ny * (x * aupyy - y * aupyx) +
             nz * (x * aupzy - y * aupzx)) * weight;
    }

    axx = chi * (axx + gxx * trk / 3.0) - trk;
    axy = chi * (axy + gxy * trk / 3.0);
    axz = chi * (axz + gxz * trk / 3.0);
    ayy = chi * (ayy + gyy * trk / 3.0) - trk;
    ayz = chi * (ayz + gyz * trk / 3.0);
    azz = chi * (azz + gzz * trk / 3.0) - trk;

    if (symmetry == 0)
    {
        output[1] += one_eighth * psi *
            (nx * axx + ny * axy + nz * axz) * weight;
        output[2] += one_eighth * psi *
            (nx * axy + ny * ayy + nz * ayz) * weight;
        output[3] += one_eighth * psi *
            (nx * axz + ny * ayz + nz * azz) * weight;
    }
    else if (symmetry == 1)
    {
        output[1] += one_eighth * psi *
            (nx * axx + ny * axy + nz * axz) * weight;
        output[2] += one_eighth * psi *
            (nx * axy + ny * ayy + nz * ayz) * weight;
    }
}

static void omp_surface_adm(
    OmpAnalysisCache *cache, double radius, int level, cgh *gh,
    var *chi, var *trk, var *gxx, var *gxy, var *gxz, var *gyy,
    var *gyz, var *gzz, var *axx, var *axy, var *axz, var *ayy,
    var *ayz, var *azz, var *gmx, var *gmy, var *gmz,
    var *sfx, var *sfy, var *sfz, double *result,
    int symmetry, int factor, int points, int n_phi, double dphi,
    const double *theta_weight, const double *nx, const double *ny,
    const double *nz)
{
    std::vector<Block *> work_blocks;
    MyList<Patch> *patch_node = gh->PatL[level];
    while (patch_node)
    {
        const std::vector<Block *> patch_blocks =
            omp_collect_blocks(patch_node->data);
        work_blocks.insert(work_blocks.end(), patch_blocks.begin(),
                           patch_blocks.end());
        patch_node = patch_node->next;
    }

#pragma omp parallel for schedule(static)
    for (std::size_t block_index = 0;
         block_index < work_blocks.size(); ++block_index)
    {
        Block *block = work_blocks[block_index];
        f_admmass_bssn(
            block->shape, block->X[0], block->X[1], block->X[2],
            block->fgfs[chi->sgfn], block->fgfs[trk->sgfn],
            block->fgfs[gxx->sgfn], block->fgfs[gxy->sgfn],
            block->fgfs[gxz->sgfn], block->fgfs[gyy->sgfn],
            block->fgfs[gyz->sgfn], block->fgfs[gzz->sgfn],
            block->fgfs[axx->sgfn], block->fgfs[axy->sgfn],
            block->fgfs[axz->sgfn], block->fgfs[ayy->sgfn],
            block->fgfs[ayz->sgfn], block->fgfs[azz->sgfn],
            block->fgfs[gmx->sgfn], block->fgfs[gmy->sgfn],
            block->fgfs[gmz->sgfn], block->fgfs[sfx->sgfn],
            block->fgfs[sfy->sgfn], block->fgfs[sfz->sgfn],
            symmetry);
    }

    OmpSpherePlan &plan = omp_get_sphere_plan(
        cache, gh->PatL[level]->data, level, radius, points,
        nx, ny, nz, symmetry);
    var *fields[17] = {
        sfx, sfy, sfz, chi, trk, gxx, gxy, gxz, gyy,
        gyz, gzz, axx, axy, axz, ayy, ayz, azz};
    const int workers = omp_get_max_threads();
    cache->adm_partials.assign(
        static_cast<std::size_t>(workers) * 7, 0.0);

#pragma omp parallel
    {
        const int worker = omp_get_thread_num();
        double *local =
            &cache->adm_partials[static_cast<std::size_t>(worker) * 7];
#pragma omp for schedule(static)
        for (int point = 0; point < points; ++point)
        {
            double value[17];
            for (int field = 0; field < 17; ++field)
                value[field] =
                    omp_cached_interpolate(plan.points[point], fields[field]);
            const int theta_index = point / n_phi;
            omp_accumulate_adm_point(
                value, radius * nx[point], radius * ny[point],
                radius * nz[point], nx[point], ny[point], nz[point],
                theta_weight[theta_index], symmetry, local);
        }
    }

    double reduced[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int worker = 0; worker < workers; ++worker)
        for (int quantity = 0; quantity < 7; ++quantity)
            reduced[quantity] +=
                cache->adm_partials[static_cast<std::size_t>(worker) * 7 +
                                    quantity];

    result[0] = reduced[0] * radius * radius * dphi * factor;
    const double vector_scale =
        radius * radius * dphi * (1.0 / M_PI) * factor;
    result[1] = reduced[1] * vector_scale;
    result[2] = reduced[2] * vector_scale;
    result[3] = reduced[3] * vector_scale;
    result[4] = reduced[4] * vector_scale;
    result[5] = reduced[5] * vector_scale;
    result[6] = reduced[6] * vector_scale;
}

} // namespace

#endif
