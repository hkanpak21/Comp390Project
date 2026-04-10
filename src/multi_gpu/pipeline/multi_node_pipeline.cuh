#pragma once
/**
 * multi_node_pipeline.cuh
 *
 * Multi-node ciphertext pipeline using MPI + intra-node CtPipeline.
 *
 * Architecture:
 *   - MPI rank 0 (master): encrypts ciphertexts, scatters to ranks via MPI
 *   - Each MPI rank: receives its batch, creates CtPipeline for local GPUs
 *   - Each rank processes its batch independently (zero inter-node communication)
 *   - MPI rank 0: gathers results
 *
 * With 4 MN5 nodes × 4 H100 GPUs = 16 GPUs total:
 *   - 128 ciphertexts / 16 GPUs = 8 per GPU
 *   - Expected speedup: ~12-14x (embarrassingly parallel)
 *
 * Data transfer: MPI_Scatter of serialized ciphertexts at start,
 *   MPI_Gather at end. InfiniBand NDR200 (200 Gb/s) handles this.
 */

#ifdef USE_MPI

#include <mpi.h>
#include <vector>
#include <functional>

#include "ct_pipeline.cuh"
#include "ciphertext.h"
#include "context.cuh"
#include "secretkey.h"

namespace nexus_multi_gpu {

class MultiNodePipeline {
public:
    // Create multi-node pipeline.
    // Must be called by ALL MPI ranks.
    static MultiNodePipeline create(
        const phantom::EncryptionParameters &parms,
        int gpus_per_node,
        PhantomSecretKey &secret_key);

    // Scatter ciphertexts from rank 0 to all ranks.
    // Only rank 0 needs to provide non-empty cts.
    void scatter(const std::vector<PhantomCiphertext> &cts);

    // Execute on each rank's local GPUs.
    // Same function signature as CtPipeline::execute.
    void execute(CtPipeline::GpuFn fn);

    // Gather results back to rank 0.
    std::vector<PhantomCiphertext> gather();

    int world_size() const { return world_size_; }
    int rank() const { return rank_; }
    int total_gpus() const { return world_size_ * gpus_per_node_; }

    void destroy();

private:
    int world_size_ = 1;
    int rank_ = 0;
    int gpus_per_node_ = 1;
    CtPipeline local_pipeline_;
    std::vector<PhantomCiphertext> local_cts_;
    size_t total_cts_ = 0;
    phantom::EncryptionParameters parms_;
};

} // namespace nexus_multi_gpu

#endif // USE_MPI
