/**
 * multi_node_pipeline.cu
 *
 * Multi-node ciphertext pipeline: MPI scatter → local CtPipeline → MPI gather.
 *
 * Serialization: PhantomCiphertext.save() / .load() for MPI transfer.
 * Each rank gets total_cts / world_size ciphertexts.
 */

#ifdef USE_MPI

#include "multi_node_pipeline.cuh"

#include <cstdio>
#include <sstream>

namespace nexus_multi_gpu {

MultiNodePipeline MultiNodePipeline::create(
    const phantom::EncryptionParameters &parms,
    int gpus_per_node,
    PhantomSecretKey &secret_key)
{
    MultiNodePipeline mnp;
    MPI_Comm_rank(MPI_COMM_WORLD, &mnp.rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mnp.world_size_);
    mnp.gpus_per_node_ = gpus_per_node;
    mnp.parms_ = parms;

    // Create local pipeline for this node's GPUs
    mnp.local_pipeline_ = CtPipeline::create(parms, gpus_per_node, secret_key);

    if (mnp.rank_ == 0) {
        printf("[MultiNodePipeline] %d nodes × %d GPUs = %d total GPUs\n",
               mnp.world_size_, gpus_per_node, mnp.total_gpus());
    }

    return mnp;
}

void MultiNodePipeline::scatter(const std::vector<PhantomCiphertext> &cts) {
    total_cts_ = cts.size();

    // Broadcast total count
    int total = (int)total_cts_;
    MPI_Bcast(&total, 1, MPI_INT, 0, MPI_COMM_WORLD);
    total_cts_ = total;

    int per_rank = total / world_size_;

    if (rank_ == 0) {
        // Serialize and send each rank's batch
        for (int r = 1; r < world_size_; r++) {
            for (int i = r * per_rank; i < (r + 1) * per_rank && i < total; i++) {
                // Serialize ciphertext
                std::stringstream ss;
                cts[i].save(ss);
                std::string data = ss.str();
                int size = (int)data.size();
                MPI_Send(&size, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(data.data(), size, MPI_CHAR, r, 1, MPI_COMM_WORLD);
            }
        }
        // Keep rank 0's batch locally
        local_cts_.clear();
        for (int i = 0; i < per_rank && i < total; i++) {
            local_cts_.push_back(cts[i]);  // shallow copy — data on GPU 0
        }
    } else {
        // Receive this rank's batch
        local_cts_.resize(per_rank);
        for (int i = 0; i < per_rank; i++) {
            int size;
            MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::string data(size, '\0');
            MPI_Recv(&data[0], size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::stringstream ss(data);
            local_cts_[i].load(ss);
        }
    }

    // Scatter to local GPUs
    local_pipeline_.scatter(local_cts_);
}

void MultiNodePipeline::execute(CtPipeline::GpuFn fn) {
    local_pipeline_.execute(fn);
}

std::vector<PhantomCiphertext> MultiNodePipeline::gather() {
    auto local_results = local_pipeline_.gather();
    int per_rank = total_cts_ / world_size_;

    if (rank_ == 0) {
        std::vector<PhantomCiphertext> all_results(total_cts_);

        // Copy rank 0's results
        for (int i = 0; i < per_rank && i < (int)local_results.size(); i++) {
            all_results[i] = std::move(local_results[i]);
        }

        // Receive from other ranks
        for (int r = 1; r < world_size_; r++) {
            for (int i = 0; i < per_rank; i++) {
                int size;
                MPI_Recv(&size, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::string data(size, '\0');
                MPI_Recv(&data[0], size, MPI_CHAR, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::stringstream ss(data);
                all_results[r * per_rank + i].load(ss);
            }
        }
        return all_results;
    } else {
        // Send results to rank 0
        for (auto &ct : local_results) {
            std::stringstream ss;
            ct.save(ss);
            std::string data = ss.str();
            int size = (int)data.size();
            MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(data.data(), size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }
        return {};
    }
}

void MultiNodePipeline::destroy() {
    local_pipeline_.destroy();
}

} // namespace nexus_multi_gpu

#endif // USE_MPI
