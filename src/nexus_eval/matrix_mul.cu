/**
 * matrix_mul.cu — Ported from vendor/nexus/cuda/src/matrix_mul.cu
 * API changes:
 *   - params_id()/parms_id() → chain_index()
 *   - global_variables::default_stream → cudaStreamPerThread
 *   - p.parms_id() = ... → p.set_chain_index(...)
 */

#include <algorithm>
#include <fstream>

#include "matrix_mul.cuh"
#include "utils.cuh"

using namespace std;
using namespace phantom::util;
using namespace phantom::arith;
using namespace nexus;

__global__ void kernel_compress_ciphertext(uint64_t *plain_data, size_t plain_scale, size_t degree,
                                           const DModulus *moduli, const double *values) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < degree) {
    auto coeffd = std::round(values[idx] * plain_scale);
    bool is_negative = std::signbit(coeffd);
    auto coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));

    if (is_negative) {
      for (std::size_t j = 0; j < 2; j++) {
        plain_data[idx + (j * degree)] = negate_uint64_mod(
            barrett_reduce_uint64_uint64(coeffu, moduli[j].value(), moduli[j].const_ratio()[1]), moduli[j].value());
      }
    } else {
      for (std::size_t j = 0; j < 2; j++) {
        plain_data[idx + (j * degree)] = barrett_reduce_uint64_uint64(coeffu, moduli[j].value(), moduli[j].const_ratio()[1]);
      }
    }
  }
}

void MMEvaluator::multiply_power_of_x(PhantomCiphertext &encrypted, PhantomCiphertext &destination, int index) {
  auto context = ckks->context;
  auto coeff_count = ckks->degree;
  // PORTED: params_id() → chain_index()
  auto param = context->get_context_data(encrypted.chain_index()).parms();
  auto moduli = param.coeff_modulus();
  auto coeff_mod_count = param.coeff_modulus().size();
  auto encrypted_count = encrypted.size();
  auto rns_coeff_count = coeff_count * coeff_mod_count;

  // PORTED: use cudaStreamPerThread instead of global_variables::default_stream
  const auto &stream = cudaStreamPerThread;

  destination = encrypted;
  ckks->evaluator.transform_from_ntt_inplace(destination);

  auto dest_data = new uint64_t[rns_coeff_count * encrypted_count];
  auto dest_data_copy = new uint64_t[rns_coeff_count * encrypted_count];
  cudaMemcpyAsync(dest_data, destination.data(), encrypted_count * rns_coeff_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::copy(dest_data, dest_data + rns_coeff_count * encrypted_count, dest_data_copy);

  for (size_t i = 0; i < encrypted_count; i++) {
    for (size_t j = 0; j < coeff_mod_count; j++) {
      uint64_t *poly = dest_data_copy + i * rns_coeff_count + j * coeff_count;
      uint64_t *result = dest_data + i * rns_coeff_count + j * coeff_count;

      uint64_t index_raw = index;
      uint64_t coeff_count_mod_mask = static_cast<uint64_t>(coeff_count) - 1;
      for (size_t k = 0; k < coeff_count; k++, poly++, index_raw++) {
        uint64_t idx = index_raw & coeff_count_mod_mask;
        if (!(index_raw & static_cast<uint64_t>(coeff_count)) || !*poly) {
          result[idx] = *poly;
        } else {
          result[idx] = moduli[j].value() - *poly;
        }
      }
    }
  }

  cudaMemcpyAsync(destination.data(), dest_data, encrypted_count * rns_coeff_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  delete[] dest_data;
  delete[] dest_data_copy;

  ckks->evaluator.transform_to_ntt_inplace(destination);
}

void MMEvaluator::enc_compress_ciphertext(vector<double> &values, PhantomCiphertext &ct) {
  size_t plain_scale = 10000000000;

  // PORTED: chain_index instead of parms_id
  auto &context_data = ckks->context->first_context_data();
  auto param = context_data.parms();
  auto moduli = ckks->context->gpu_rns_tables().modulus();
  auto coeff_modulus_size = param.coeff_modulus().size();
  auto poly_modulus_degree = param.poly_modulus_degree();

  // PORTED: use cudaStreamPerThread instead of global_variables::default_stream
  const auto &stream = cudaStreamPerThread;

  PhantomPlaintext p;
  p.resize(coeff_modulus_size, poly_modulus_degree, stream);

  auto gpu_values = make_cuda_auto_ptr<double>(values.size(), stream);
  cudaMemcpyAsync(gpu_values.get(), values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice, stream);

  kernel_compress_ciphertext<<<poly_modulus_degree / blockDimGlb.x, blockDimGlb, 0, stream>>>(
      p.data(), plain_scale, poly_modulus_degree, moduli, gpu_values.get());

  // Transform polynomials to the NTT domain
  nwt_2d_radix8_forward_inplace(p.data(), ckks->context->gpu_rns_tables(), coeff_modulus_size, 0, stream);

  // PORTED: parms_id() = ... → set_chain_index(...)
  p.set_chain_index(context_data.chain_index());
  // Plaintext doesn't have set_scale, use const_cast workaround
  const_cast<double&>(p.scale()) = plain_scale;

  // Create a ciphertext encrypting zero
  PhantomPlaintext zero_pt;
  PhantomCiphertext zero;
  ckks->encoder.encode(0.0, plain_scale, zero_pt);
  ckks->encryptor.encrypt(zero_pt, zero);

  // Encrypt the plaintext
  ckks->evaluator.add_plain(zero, p, ct);
}

vector<PhantomCiphertext> MMEvaluator::decompress_ciphertext(PhantomCiphertext &encrypted) {
  auto N = ckks->degree;
  uint32_t logN = ceil(log2(N));

  vector<PhantomCiphertext> temp;
  temp.push_back(encrypted);

  PhantomCiphertext tempctxt_rotated;
  PhantomCiphertext tempctxt_shifted;
  PhantomCiphertext tempctxt_rotatedshifted;

  for (uint32_t i = 0; i < logN; i++) {
    vector<PhantomCiphertext> newtemp(temp.size() << 1);

    uint32_t galois_elt = ckks->galois_elts[i];
    int index_raw = (N << 1) - (1 << i);
    int index = (index_raw * galois_elt) % (N << 1);

    for (uint32_t a = 0; a < temp.size(); a++) {
      ckks->evaluator.apply_galois(temp[a], galois_elt, *(ckks->galois_keys), tempctxt_rotated);
      ckks->evaluator.add(temp[a], tempctxt_rotated, newtemp[a]);
      multiply_power_of_x(temp[a], tempctxt_shifted, index_raw);
      multiply_power_of_x(tempctxt_rotated, tempctxt_rotatedshifted, index);
      ckks->evaluator.add(tempctxt_shifted, tempctxt_rotatedshifted, newtemp[a + temp.size()]);
    }

    temp = newtemp;
  }

  return temp;
}

void MMEvaluator::matrix_mul(vector<vector<double>> &x, vector<vector<double>> &y, vector<PhantomCiphertext> &res) {
  // Full 64-column path. Implemented as a thin wrapper over the range
  // variant so the original code path keeps the same observable behaviour,
  // and the per-column inner loop has only one definition to audit.
  matrix_mul_range(x, y, res, 0, 64);
}

void MMEvaluator::matrix_mul_range(vector<vector<double>> &x,
                                   vector<vector<double>> &y,
                                   vector<PhantomCiphertext> &res,
                                   int cols_lo,
                                   int cols_hi) {
  // Output-channel split MatMul. Computes only columns [cols_lo, cols_hi)
  // of the 64-column NEXUS MatMul. To make this a real (per-thread, per-GPU)
  // speedup vs single-GPU full-range, BOTH the dominant per-column compute
  // AND the shared decompress setup are restricted to the column range:
  //
  //   • Each output column i uses b_expanded_cts[i*768 + j] for j in [0,768)
  //   • Each b_compressed_cts[k] decompresses into N expanded ciphertexts
  //     [k*N, (k+1)*N) for N = ckks->degree (8192 at logN=13)
  //   • Therefore column i lives in compressed indices [i*768/N, ((i+1)*768)/N]
  //   • For [cols_lo, cols_hi) we only decompress compressed indices in the
  //     union of these ranges — at logN=13 with 64 cols across 6 compressed
  //     this is typically 2–3 of the 6 (a 2-3× setup reduction)
  //
  // Compress is cheap (a few ms total), so we run the full compress for all
  // 6 b_cts to keep the index math identical to single-GPU. Only decompress
  // is restricted.
  if (cols_lo < 0)  cols_lo = 0;
  if (cols_hi > 64) cols_hi = 64;
  if (cols_lo >= cols_hi) {
    res.clear();
    return;
  }

  auto timer = Timer();

  // Encode plaintext
  vector<PhantomPlaintext> a_pts;
  a_pts.reserve(768);

  for (int i = 0; i < 768; i++) {
    PhantomPlaintext pt;
    ckks->encoder.encode(x[i], ckks->scale, pt);
    a_pts.push_back(pt);
  }

  // Ciphertext encoding & compression (cheap; same as single-GPU full path).
  timer.start();

  size_t N_degree = ckks->degree;
  int b_cts_count = 768 * 64 / (int)N_degree;
  vector<PhantomCiphertext> b_compressed_cts;
  b_compressed_cts.reserve(b_cts_count);

  for (int i = 0; i < b_cts_count; i++) {
    PhantomCiphertext ct;
    enc_compress_ciphertext(y[i], ct);
    b_compressed_cts.push_back(ct);
  }

  timer.stop();
  cout << "Compression took: " << timer.duration<chrono::milliseconds>() << " milliseconds" << endl;

  // Ciphertext decompression — RESTRICTED to compressed indices needed by
  // the [cols_lo, cols_hi) slice. Index k of compressed[k] expands to the
  // 768*64/N_degree-aligned chunk of expanded indices [k*N_degree, (k+1)*N_degree),
  // so we pick the [k_lo, k_hi) range that covers expanded indices
  // [cols_lo*768, cols_hi*768).
  timer.start();

  int exp_lo = cols_lo * 768;
  int exp_hi = cols_hi * 768;
  int k_lo = exp_lo / (int)N_degree;
  int k_hi = (exp_hi + (int)N_degree - 1) / (int)N_degree;
  if (k_hi > b_cts_count) k_hi = b_cts_count;

  // Sparse map from expanded-index → ciphertext (vector of length 64*768
  // = 49152 at logN=13 worst case, but only the slice we decompressed is
  // populated). We keep it dense for index-symmetry with the single-GPU
  // path so the per-column inner loop reads `b_expanded_cts[i*768 + j]`
  // exactly as before.
  vector<PhantomCiphertext> b_expanded_cts(64 * 768);

  for (int k = k_lo; k < k_hi; k++) {
    vector<PhantomCiphertext> temp_cts = decompress_ciphertext(b_compressed_cts[k]);
    cout << "Expanded ciphertext #" << k + 1 << " (slice)" << endl;
    int dst = k * (int)N_degree;
    for (size_t t = 0; t < temp_cts.size() && (size_t)(dst + (int)t) < b_expanded_cts.size(); t++) {
      b_expanded_cts[dst + (int)t] = std::move(temp_cts[t]);
    }
  }

  timer.stop();
  cout << "Decompression took: " << timer.duration<chrono::seconds>() << " seconds"
       << " (k=[" << k_lo << "," << k_hi << ") of " << b_cts_count << ")" << endl;

  // Perform plain-cipher matrix multiplication restricted to [cols_lo, cols_hi).
  timer.start();

  res.clear();
  res.reserve(cols_hi - cols_lo);

  for (int i = cols_lo; i < cols_hi; i++) {
    PhantomCiphertext res_col_ct;
    vector<PhantomCiphertext> temp_cts(768);

    for (int j = 0; j < 768; j++) {
      ckks->evaluator.multiply_plain(b_expanded_cts[i * 768 + j], a_pts[j], temp_cts[j]);
    }

    res_col_ct.set_scale(temp_cts[0].scale());
    ckks->evaluator.add_many(temp_cts, res_col_ct);

    res_col_ct.set_scale(res_col_ct.scale() * 4096);
    res.push_back(res_col_ct);
  }

  for (auto &ct : res) {
    while (ct.coeff_modulus_size() > 1) {
      ckks->evaluator.rescale_to_next_inplace(ct);
    }
  }

  timer.stop();
  cout << "Result calculation time: " << timer.duration<chrono::milliseconds>() << " milliseconds"
       << " (cols [" << cols_lo << "," << cols_hi << "))" << endl;
}

void MMEvaluator::matrix_mul_unified(
    vector<PhantomCiphertext> &x_cts,
    vector<vector<double>> &weights,
    int n_columns,
    vector<PhantomCiphertext> &res)
{
  // Unified MatMul at N=65536 — no compress/decompress needed.
  // Each x_cts[i] holds one encrypted input vector (e.g., 768 values in 32768 slots).
  // weights[j] holds one plaintext weight row (same slot count).
  // For each output column i: result = sum_j(x_cts[i] * weights[j])
  //
  // This is the same mathematical operation as the original matrix_mul()
  // but without the N=8192 ciphertext packing optimization.

  int inner_dim = (int)weights.size();

  // Encode weight plaintexts at the same chain_index as the input ciphertexts
  size_t ct_chain_idx = x_cts[0].chain_index();
  vector<PhantomPlaintext> weight_pts(inner_dim);
  for (int j = 0; j < inner_dim; j++) {
    ckks->encoder.encode(weights[j], ct_chain_idx, ckks->scale, weight_pts[j]);
  }

  res.clear();
  res.reserve(n_columns);

  for (int i = 0; i < n_columns; i++) {
    // inner_dim multiply_plain operations + add accumulation
    vector<PhantomCiphertext> temp_cts(inner_dim);
    for (int j = 0; j < inner_dim; j++) {
      ckks->evaluator.multiply_plain(x_cts[i], weight_pts[j], temp_cts[j]);
    }

    // Sum all terms
    PhantomCiphertext acc;
    ckks->evaluator.add_many(temp_cts, acc);

    // Rescale (consumes 1 level)
    ckks->evaluator.rescale_to_next_inplace(acc);

    res.push_back(std::move(acc));
  }
}
