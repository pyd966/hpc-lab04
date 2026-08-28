#ifndef AMSS_OMP_ONLY_MPI_H
#define AMSS_OMP_ONLY_MPI_H

// The CPU OpenMP-only build still includes mpi.h for the existing public
// function signatures, but it does not initialize or launch an MPI world.
// These single-rank shims keep the source compatible while turning collectives
// into local copies and point-to-point calls into no-ops.
#include <mpi.h>
#include <omp.h>

#include <cstdlib>
#include <cstring>
#include <unistd.h>

static int amss_omp_world_handle;
static int amss_omp_double_handle;
static int amss_omp_int_handle;
static int amss_omp_double_int_handle;
static int amss_omp_sum_handle;
static int amss_omp_maxloc_handle;

#undef MPI_COMM_WORLD
#undef MPI_DOUBLE
#undef MPI_INT
#undef MPI_DOUBLE_INT
#undef MPI_SUM
#undef MPI_MAXLOC
#define MPI_COMM_WORLD ((MPI_Comm)&amss_omp_world_handle)
#define MPI_DOUBLE ((MPI_Datatype)&amss_omp_double_handle)
#define MPI_INT ((MPI_Datatype)&amss_omp_int_handle)
#define MPI_DOUBLE_INT ((MPI_Datatype)&amss_omp_double_int_handle)
#define MPI_SUM ((MPI_Op)&amss_omp_sum_handle)
#define MPI_MAXLOC ((MPI_Op)&amss_omp_maxloc_handle)

namespace amss_omp_only {

inline int init(int *, char ***) { return 0; }
inline int finalize() { return 0; }
inline int comm_size(MPI_Comm, int *size) {
    *size = 1;
    return 0;
}
inline int comm_rank(MPI_Comm, int *rank) {
    *rank = 0;
    return 0;
}
inline double wtime() { return omp_get_wtime(); }
inline int abort(MPI_Comm, int code) {
    std::exit(code);
}

inline std::size_t datatype_size(MPI_Datatype datatype) {
    if (datatype == MPI_DOUBLE)
        return sizeof(double);
    if (datatype == MPI_INT)
        return sizeof(int);
    if (datatype == MPI_DOUBLE_INT) {
        struct double_int_pair {
            double value;
            int rank;
        };
        return sizeof(double_int_pair);
    }
    return 1;
}

inline int allreduce(const void *sendbuf, void *recvbuf, int count,
                     MPI_Datatype datatype, MPI_Op, MPI_Comm) {
    if (sendbuf == MPI_IN_PLACE || recvbuf == sendbuf)
        return 0;
    std::memcpy(recvbuf, sendbuf, datatype_size(datatype) * static_cast<std::size_t>(count));
    return 0;
}

inline int bcast(void *, int, MPI_Datatype, int, MPI_Comm) { return 0; }
inline int isend(const void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *) { return 0; }
inline int irecv(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *) { return 0; }
inline int waitall(int, MPI_Request *, MPI_Status *) { return 0; }
inline int send(const void *, int, MPI_Datatype, int, int, MPI_Comm) { return 0; }
inline int recv(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *) { return 0; }

inline int get_processor_name(char *name, int *length) {
    if (gethostname(name, MPI_MAX_PROCESSOR_NAME) != 0)
        name[0] = '\0';
    name[MPI_MAX_PROCESSOR_NAME - 1] = '\0';
    *length = static_cast<int>(std::strlen(name));
    return 0;
}

} // namespace amss_omp_only

#define MPI_Init(argc, argv) amss_omp_only::init((argc), (argv))
#define MPI_Finalize() amss_omp_only::finalize()
#define MPI_Comm_size(comm, size) amss_omp_only::comm_size((comm), (size))
#define MPI_Comm_rank(comm, rank) amss_omp_only::comm_rank((comm), (rank))
#define MPI_Wtime() amss_omp_only::wtime()
#define MPI_Abort(comm, code) amss_omp_only::abort((comm), (code))
#define MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm) \
    amss_omp_only::allreduce((sendbuf), (recvbuf), (count), (datatype), (op), (comm))
#define MPI_Bcast(buffer, count, datatype, root, comm) \
    amss_omp_only::bcast((buffer), (count), (datatype), (root), (comm))
#define MPI_Isend(buffer, count, datatype, dest, tag, comm, request) \
    amss_omp_only::isend((buffer), (count), (datatype), (dest), (tag), (comm), (request))
#define MPI_Irecv(buffer, count, datatype, source, tag, comm, request) \
    amss_omp_only::irecv((buffer), (count), (datatype), (source), (tag), (comm), (request))
#define MPI_Waitall(count, requests, statuses) \
    amss_omp_only::waitall((count), (requests), (statuses))
#define MPI_Send(buffer, count, datatype, dest, tag, comm) \
    amss_omp_only::send((buffer), (count), (datatype), (dest), (tag), (comm))
#define MPI_Recv(buffer, count, datatype, source, tag, comm, status) \
    amss_omp_only::recv((buffer), (count), (datatype), (source), (tag), (comm), (status))
#define MPI_Get_processor_name(name, length) \
    amss_omp_only::get_processor_name((name), (length))

#endif
