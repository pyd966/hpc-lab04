cmake_minimum_required(VERSION 3.20)
project(amss_ncku_lab LANGUAGES CXX Fortran)

# --- Options ----------------------------------------------------------------
option(AMSS_ENABLE_GPU "Enable CUDA GPU build (ABEGPU target)" OFF)
set(AMSS_OPT "-O3" CACHE STRING "C/C++/Fortran optimization level")
set(AMSS_ARCH_FLAGS "" CACHE STRING "Architecture-specific compiler flags")
set(AMSS_TWOPUNCTURE_OPT "-O3" CACHE STRING "TwoPuncture-only optimization level")
set(AMSS_TWOPUNCTURE_ARCH_FLAGS "-march=native" CACHE STRING "TwoPuncture-only architecture flags")
option(AMSS_ENABLE_OPENMP "Link OpenMP to all executables" OFF)
option(AMSS_ENABLE_OMP_ONLY "Build the CPU ABE target as one OpenMP-only process" OFF)
option(AMSS_ENABLE_OMP_DIAGNOSTICS "Collect CPU ABE OpenMP phase utilization diagnostics" OFF)
option(AMSS_ENABLE_OMP_CONSTRAINT_PARALLEL "Parallelize CPU ABE constraint recomputation across blocks" ON)
option(AMSS_ENABLE_OMP_INITIAL_CONSTRAINT_PARALLEL "Parallelize CPU ABE initial constraint recomputation across blocks" ON)
option(AMSS_ENABLE_KODIS_SIMD "Vectorize the CPU ABE Kreiss-Oliger dissipation loop" ON)
option(AMSS_ENABLE_FDERIVS_SIMD "Vectorize the CPU ABE regular fderivs interior loop" ON)
option(AMSS_ENABLE_FDERIVS_BOUNDARY_INIT "Zero only uncovered fderivs boundary planes" OFF)
option(AMSS_ENABLE_FDERIVS_BATCH "Batch pairs of regular fderivs fields" OFF)
option(AMSS_ENABLE_FDDERIVS_SIMD "Vectorize the CPU ABE regular fdderivs interior loop" ON)
option(AMSS_ENABLE_FDDERIVS_BOUNDARY_INIT "Zero only uncovered fdderivs boundary planes" OFF)
option(AMSS_ENABLE_OMP_DIRECT_SYNC "Bypass the pack workspace for provably disjoint same-level Sync transfers" OFF)
option(AMSS_ENABLE_OMP_DIRECT_AMR_TRANSFER "Write non-mixed prolong/restrict results directly to their destination arrays" OFF)
option(AMSS_ENABLE_OMP_DIRECT_AMR_SPLIT "Split non-overlapping direct AMR target regions into separate OpenMP tasks" OFF)
option(AMSS_ENABLE_LOPSIDEDIFF_SIMD "Vectorize the CPU ABE regular lopsidediff interior loop" ON)
option(AMSS_ENABLE_LOPSIDEDIFF_GHOST_ONLY "Avoid full lopsided ghost copies in the deep interior" OFF)
option(AMSS_ENABLE_SYMMETRY_BD_NO_CLEAR "Skip redundant full ghost-array zeroing in symmetry_bd" ON)
option(AMSS_ENABLE_LOPSIDEDIFF_BATCH "Batch pairs of lopsided advection fields" OFF)
option(AMSS_ENABLE_RHS_METRIC_FUSION "Fuse the CPU ABE metric initialization and RHS passes" ON)
option(AMSS_ENABLE_RHS_GAMMA_FUSION "Fuse the CPU ABE Gamma RHS passes" ON)
option(AMSS_ENABLE_RHS_SANITY_CHECK "Run the CPU ABE per-RHS NaN sanity scan" OFF)
option(AMSS_ENABLE_TWOPUNCTURE_OPENMP "Enable OpenMP for TwoPunctureABE" ON)
set(AMSS_MPI_CUDA_AWARE 0 CACHE STRING "Set MPI_CUDA_AWARE macro for ABEGPU (0/1)")
set_property(CACHE AMSS_MPI_CUDA_AWARE PROPERTY STRINGS 0 1)

# --- Compile flags ----------------------------------------------------------
# Legacy AMSS-NCKU has strict-aliasing UB; -fno-strict-aliasing is kept
# regardless of AMSS_OPT so changing -O does not lose the safety flag.
# `-Dfortran3 -Dnewc` select the modern C/Fortran source branches (without
# them the C++ side falls back to pre-ANSI <iostream.h> includes and the
# Fortran side skips the f90 modules). Apply to every supported compiler.
separate_arguments(AMSS_OPT_LIST UNIX_COMMAND "${AMSS_OPT}")
separate_arguments(AMSS_ARCH_FLAGS_LIST UNIX_COMMAND "${AMSS_ARCH_FLAGS}")
separate_arguments(AMSS_TWOPUNCTURE_OPT_LIST UNIX_COMMAND "${AMSS_TWOPUNCTURE_OPT}")
separate_arguments(AMSS_TWOPUNCTURE_ARCH_FLAGS_LIST UNIX_COMMAND "${AMSS_TWOPUNCTURE_ARCH_FLAGS}")

# Clang/IntelLLVM/ARMClang all accept the GCC-style flags used here.
# COMPILE_LANG_AND_ID takes comma-separated compiler IDs (not CMake list).
add_compile_options(
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,IntelLLVM,Intel,ARMClang>:${AMSS_OPT_LIST};${AMSS_ARCH_FLAGS_LIST};-fno-strict-aliasing;-Wno-deprecated;-Dfortran3;-Dnewc>"
  "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,Flang,IntelLLVM,Intel,ARMClang>:${AMSS_OPT_LIST};${AMSS_ARCH_FLAGS_LIST};-fno-strict-aliasing;-cpp>"
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- MPI --------------------------------------------------------------------
find_package(MPI REQUIRED COMPONENTS CXX)

if(AMSS_ENABLE_OMP_ONLY)
  set(AMSS_ENABLE_OPENMP ON CACHE BOOL "Link OpenMP to all executables" FORCE)
endif()

# --- Shared Fortran sources -------------------------------------------------
set(AMSS_F90SOURCES
  src/bssn_rhs.f90
  src/diff_new.f90 src/enforce_algebra.f90
  src/fadmquantites_bssn.f90 src/fmisc.f90
  src/getnp4.f90
  src/initial_puncture.f90 src/kodiss.f90
  src/lopsidediff.f90 src/lopsided_batch.f90 src/symmetry_partial.f90
  src/prolongrestrict_cell.f90 src/rungekutta4_rout.f90
  src/sommerfeld_rout.f90
)

# --- ABE: CPU main evolution executable -------------------------------------
set(ABE_SOURCES
  src/ABE.C src/Ansorg.C src/Block.C src/MPatch.C src/Parallel.C
  src/bssn_class.C src/cgh.C src/checkpoint.C
  src/misc.C src/monitor.C src/perf.C
  src/surface_integral.C src/transpbh.C src/var.C
  ${AMSS_F90SOURCES}
)
add_executable(ABE ${ABE_SOURCES})
if(AMSS_ENABLE_OMP_DIAGNOSTICS)
  target_compile_definitions(ABE PRIVATE AMSS_OMP_DIAGNOSTICS)
endif()
if(AMSS_ENABLE_OMP_CONSTRAINT_PARALLEL)
  target_compile_definitions(ABE PRIVATE AMSS_OMP_CONSTRAINT_PARALLEL)
endif()
if(AMSS_ENABLE_OMP_INITIAL_CONSTRAINT_PARALLEL)
  target_compile_definitions(ABE PRIVATE AMSS_OMP_INITIAL_CONSTRAINT_PARALLEL)
endif()
if(AMSS_ENABLE_KODIS_SIMD)
  target_compile_definitions(ABE PRIVATE AMSS_KODIS_SIMD)
endif()
if(AMSS_ENABLE_FDERIVS_SIMD)
  target_compile_definitions(ABE PRIVATE AMSS_FDERIVS_SIMD)
endif()
if(AMSS_ENABLE_FDERIVS_BATCH)
  target_compile_definitions(ABE PRIVATE AMSS_FDERIVS_BATCH)
endif()
if(AMSS_ENABLE_FDDERIVS_SIMD)
  target_compile_definitions(ABE PRIVATE AMSS_FDDERIVS_SIMD)
endif()
if(AMSS_ENABLE_FDERIVS_BOUNDARY_INIT)
  target_compile_definitions(ABE PRIVATE AMSS_FDERIVS_BOUNDARY_INIT)
endif()
if(AMSS_ENABLE_FDDERIVS_BOUNDARY_INIT)
  target_compile_definitions(ABE PRIVATE AMSS_FDDERIVS_BOUNDARY_INIT)
endif()
if(AMSS_ENABLE_OMP_DIRECT_SYNC)
  target_compile_definitions(ABE PRIVATE AMSS_OMP_DIRECT_SYNC)
endif()
if(AMSS_ENABLE_OMP_DIRECT_AMR_TRANSFER)
  target_compile_definitions(ABE PRIVATE AMSS_OMP_DIRECT_AMR_TRANSFER)
endif()
if(AMSS_ENABLE_OMP_DIRECT_AMR_SPLIT)
  target_compile_definitions(ABE PRIVATE AMSS_OMP_DIRECT_AMR_SPLIT)
endif()
if(AMSS_ENABLE_LOPSIDEDIFF_SIMD)
  target_compile_definitions(ABE PRIVATE AMSS_LOPSIDEDIFF_SIMD)
endif()
if(AMSS_ENABLE_LOPSIDEDIFF_GHOST_ONLY)
  target_compile_definitions(ABE PRIVATE AMSS_LOPSIDEDIFF_GHOST_ONLY)
endif()
if(AMSS_ENABLE_LOPSIDEDIFF_BATCH)
  target_compile_definitions(ABE PRIVATE AMSS_ENABLE_LOPSIDEDIFF_BATCH)
endif()
if(AMSS_ENABLE_SYMMETRY_BD_NO_CLEAR)
  target_compile_definitions(ABE PRIVATE AMSS_SYMMETRY_BD_NO_CLEAR)
endif()
if(AMSS_ENABLE_RHS_METRIC_FUSION)
  target_compile_definitions(ABE PRIVATE AMSS_ENABLE_RHS_METRIC_FUSION)
endif()
if(AMSS_ENABLE_RHS_GAMMA_FUSION)
  target_compile_definitions(ABE PRIVATE AMSS_ENABLE_RHS_GAMMA_FUSION)
endif()
if(AMSS_ENABLE_RHS_SANITY_CHECK)
  target_compile_definitions(ABE PRIVATE AMSS_ENABLE_RHS_SANITY_CHECK)
endif()
if(AMSS_ENABLE_OMP_ONLY)
  target_compile_definitions(ABE PRIVATE AMSS_OMP_ONLY)
  # Keep MPI public type declarations for the compatibility shim without linking MPI runtime.
  target_include_directories(ABE PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_compile_options(ABE PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-include${CMAKE_CURRENT_SOURCE_DIR}/src/omp_only_mpi.h>"
  )
else()
  target_link_libraries(ABE PRIVATE MPI::MPI_CXX)
endif()

# --- TwoPunctureABE: initial-data solver -----------------------------------
add_executable(TwoPunctureABE src/TwoPunctureABE.C src/TwoPunctures.C)
target_compile_options(TwoPunctureABE PRIVATE
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,IntelLLVM,Intel,ARMClang>:${AMSS_TWOPUNCTURE_OPT_LIST};${AMSS_TWOPUNCTURE_ARCH_FLAGS_LIST}>"
)
target_link_libraries(TwoPunctureABE PRIVATE MPI::MPI_CXX)

# --- Optional OpenMP --------------------------------------------------------
if(AMSS_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX Fortran)
  target_link_libraries(ABE PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_Fortran)
  target_link_libraries(TwoPunctureABE PRIVATE OpenMP::OpenMP_CXX)
elseif(AMSS_ENABLE_TWOPUNCTURE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(TwoPunctureABE PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- ABEGPU: GPU evolution executable (device-resident baseline) -----------
# Same C++/Fortran sources as ABE except bssn_class.C -> bssn_gpu_class.C,
# plus GPU kernel/agent files.
set(ABEGPU_CXX_SOURCES
  src/ABE.C src/Ansorg.C src/Block.C src/MPatch.C src/Parallel.C
  src/bssn_gpu_class.C src/bssn_step_gpu.C
  src/cgh.C src/checkpoint.C src/misc.C
  src/monitor.C src/perf.C src/surface_integral.C
  src/transpbh.C src/var.C
  src/helper.cpp src/Parallel_GPU.cpp
)
set(ABEGPU_CUDA_SOURCES
  src/fmisc_gpu.cu src/lopsidediff_gpu.cu src/kodiss_gpu.cu src/diff_new_gpu.cu
  src/bssn_rhs_gpu.cu src/prolongrestrict_cell_gpu.cu src/gpu_manager.cu
  src/enforce_algebra_gpu.cu src/sommerfeld_rout_gpu.cu src/rungekutta4_rout_gpu.cu
  src/fadmquantites_bssn_gpu.cu src/misc_gpu.cu src/MPatch_gpu.cu
  src/surface_integral_gpu.cu src/getnp4_gpu.cu
)

if(AMSS_ENABLE_GPU)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures" FORCE)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  find_package(CUDAToolkit REQUIRED)

  add_executable(ABEGPU ${ABEGPU_CXX_SOURCES} ${ABEGPU_CUDA_SOURCES} ${AMSS_F90SOURCES})
  set_source_files_properties(${ABEGPU_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
  set_target_properties(ABEGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(ABEGPU PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-rdc=true;-lineinfo>")
  target_compile_definitions(ABEGPU PRIVATE USE_GPU fortran3 newc MPI_CUDA_AWARE=${AMSS_MPI_CUDA_AWARE})
  target_include_directories(ABEGPU PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(ABEGPU PRIVATE CUDA::cudart MPI::MPI_CXX)
  if(AMSS_ENABLE_OPENMP)
    target_link_libraries(ABEGPU PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_Fortran)
  endif()
endif()

# --- Configure summary ------------------------------------------------------
message(STATUS "==================== AMSS-NCKU Lab 4 ====================")
message(STATUS "CXX      : ${CMAKE_CXX_COMPILER}")
message(STATUS "Fortran  : ${CMAKE_Fortran_COMPILER}")
if(AMSS_ENABLE_GPU)
  message(STATUS "GPU      : ON, CUDA=${CMAKE_CUDA_COMPILER}, arch=${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "GPU      : OFF")
endif()
message(STATUS "AMSS_OPT=${AMSS_OPT} ARCH_FLAGS=${AMSS_ARCH_FLAGS} OPENMP=${AMSS_ENABLE_OPENMP} OMP_ONLY=${AMSS_ENABLE_OMP_ONLY} OMP_DIAGNOSTICS=${AMSS_ENABLE_OMP_DIAGNOSTICS} OMP_CONSTRAINT_PARALLEL=${AMSS_ENABLE_OMP_CONSTRAINT_PARALLEL} OMP_INITIAL_CONSTRAINT_PARALLEL=${AMSS_ENABLE_OMP_INITIAL_CONSTRAINT_PARALLEL} OMP_DIRECT_SYNC=${AMSS_ENABLE_OMP_DIRECT_SYNC} OMP_DIRECT_AMR_TRANSFER=${AMSS_ENABLE_OMP_DIRECT_AMR_TRANSFER} OMP_DIRECT_AMR_SPLIT=${AMSS_ENABLE_OMP_DIRECT_AMR_SPLIT} KODIS_SIMD=${AMSS_ENABLE_KODIS_SIMD} FDERIVS_SIMD=${AMSS_ENABLE_FDERIVS_SIMD} FDDERIVS_SIMD=${AMSS_ENABLE_FDDERIVS_SIMD} LOPSIDEDIFF_SIMD=${AMSS_ENABLE_LOPSIDEDIFF_SIMD} SYMMETRY_BD_NO_CLEAR=${AMSS_ENABLE_SYMMETRY_BD_NO_CLEAR} LOPSIDEDIFF_GHOST_ONLY=${AMSS_ENABLE_LOPSIDEDIFF_GHOST_ONLY} LOPSIDEDIFF_BATCH=${AMSS_ENABLE_LOPSIDEDIFF_BATCH} RHS_METRIC_FUSION=${AMSS_ENABLE_RHS_METRIC_FUSION} RHS_GAMMA_FUSION=${AMSS_ENABLE_RHS_GAMMA_FUSION} RHS_SANITY_CHECK=${AMSS_ENABLE_RHS_SANITY_CHECK} MPI_CUDA_AWARE=${AMSS_MPI_CUDA_AWARE}")
message(STATUS "TWOP_OPT=${AMSS_TWOPUNCTURE_OPT} TWOP_ARCH=${AMSS_TWOPUNCTURE_ARCH_FLAGS}")
message(STATUS "TWOP_OPENMP=${AMSS_ENABLE_TWOPUNCTURE_OPENMP}")
message(STATUS "==========================================================")
