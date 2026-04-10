/**
 * ct_pipeline.cu
 *
 * Ciphertext-level pipeline parallelism: distribute independent ciphertexts
 * across GPUs for embarrassingly parallel FHE operations.
 */

#include "ct_pipeline.cuh"

#include <cstdio>
#include <cassert>
#include <sstream>

#define CUDA_CHECK(cmd) do {                                             \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(e));                                   \
        throw std::runtime_error(cudaGetErrorString(e));                 \
    }                                                                    \
} while (0)

using namespace phantom;

namespace nexus_multi_gpu {

CtPipeline CtPipeline::create(
    const EncryptionParameters &parms,
    int n_gpus,
    PhantomSecretKey &secret_key)
{
    CtPipeline pipe;
    pipe.n_gpus_ = n_gpus;
    pipe.parms_ = parms;
    pipe.states_.resize(n_gpus);

    // Serialize secret key from GPU 0
    std::stringstream sk_buf;
    cudaSetDevice(0);
    secret_key.save(sk_buf);

    // Enable peer access
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < n_gpus; j++) {
            if (i != j) {
                int can = 0;
                cudaDeviceCanAccessPeer(&can, i, j);
                if (can) cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    // Create per-GPU PhantomContext + keys
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        auto &st = pipe.states_[g];
        st.ctx = new PhantomContext(parms);

        if (g == 0) {
            // GPU 0 uses the original secret key
            st.sk = new PhantomSecretKey();
            sk_buf.seekg(0);
            st.sk->load(sk_buf);
        } else {
            // Other GPUs load from serialized
            sk_buf.seekg(0);
            st.sk = new PhantomSecretKey();
            st.sk->load(sk_buf);
        }

        st.rk = new PhantomRelinKey(st.sk->gen_relinkey(*st.ctx));
        st.enc = new PhantomCKKSEncoder(*st.ctx);
    }

    cudaSetDevice(0);
    return pipe;
}

void CtPipeline::scatter(const std::vector<PhantomCiphertext> &cts) {
    total_cts_ = cts.size();

    // Clear previous batch
    for (auto &st : states_) {
        st.local_cts.clear();
        st.original_indices.clear();
    }

    // Round-robin assignment
    for (size_t i = 0; i < cts.size(); i++) {
        int target_gpu = i % n_gpus_;
        states_[target_gpu].original_indices.push_back(i);
    }

    // Copy ciphertexts to target GPUs
    for (int g = 0; g < n_gpus_; g++) {
        cudaSetDevice(g);
        auto &st = states_[g];

        for (size_t idx : st.original_indices) {
            const auto &src_ct = cts[idx];

            // Create a new ciphertext on this GPU
            PhantomCiphertext local_ct;
            local_ct.resize(*st.ctx, src_ct.chain_index(), src_ct.size(), cudaStreamPerThread);
            local_ct.set_scale(src_ct.scale());
            local_ct.set_ntt_form(src_ct.is_ntt_form());

            // Copy data from GPU 0 to this GPU
            size_t data_bytes = src_ct.size() * src_ct.coeff_modulus_size()
                              * src_ct.poly_modulus_degree() * sizeof(uint64_t);
            CUDA_CHECK(cudaMemcpyPeer(local_ct.data(), g, src_ct.data(), 0, data_bytes));

            st.local_cts.push_back(std::move(local_ct));
        }
    }
    cudaSetDevice(0);
}

void CtPipeline::execute(GpuFn fn) {
    std::vector<std::thread> threads;

    for (int g = 0; g < n_gpus_; g++) {
        threads.emplace_back([this, g, &fn]() {
            cudaSetDevice(g);
            auto &st = states_[g];
            fn(g, *st.ctx, *st.rk, *st.enc, st.local_cts);
            cudaDeviceSynchronize();
        });
    }

    for (auto &t : threads) t.join();
}

std::vector<PhantomCiphertext> CtPipeline::gather() {
    std::vector<PhantomCiphertext> result(total_cts_);

    cudaSetDevice(0);
    auto &st0 = states_[0];

    for (int g = 0; g < n_gpus_; g++) {
        auto &st = states_[g];

        for (size_t local_idx = 0; local_idx < st.local_cts.size(); local_idx++) {
            size_t global_idx = st.original_indices[local_idx];
            auto &src_ct = st.local_cts[local_idx];

            // Create result ciphertext on GPU 0
            PhantomCiphertext dst_ct;
            dst_ct.resize(*st0.ctx, src_ct.chain_index(), src_ct.size(), cudaStreamPerThread);
            dst_ct.set_scale(src_ct.scale());
            dst_ct.set_ntt_form(src_ct.is_ntt_form());

            size_t data_bytes = src_ct.size() * src_ct.coeff_modulus_size()
                              * src_ct.poly_modulus_degree() * sizeof(uint64_t);
            CUDA_CHECK(cudaMemcpyPeer(dst_ct.data(), 0, src_ct.data(), g, data_bytes));

            result[global_idx] = std::move(dst_ct);
        }
    }

    cudaDeviceSynchronize();
    return result;
}

void CtPipeline::destroy() {
    for (auto &st : states_) {
        delete st.enc;
        delete st.rk;
        delete st.sk;
        delete st.ctx;
        st.local_cts.clear();
    }
}

} // namespace nexus_multi_gpu
