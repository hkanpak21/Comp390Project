/**
 * softmax.cu — Ported from vendor/nexus/cuda/src/softmax.cu
 * API changes: params_id() → chain_index()
 */

#include <iostream>
#include "softmax.cuh"

using namespace nexus;

void SoftmaxEvaluator::softmax(PhantomCiphertext &x, PhantomCiphertext &res, int len) {
  PhantomCiphertext tmp, exp_x;

  int log_step = log2(len);
  ckks->evaluator.rotate_vector(x, -len, *ckks->galois_keys, tmp);
  ckks->evaluator.add_inplace(x, tmp);

  exp_x = ckks->exp(x);

  tmp = exp_x;
  for (int i = 0; i < log_step; ++i) {
    ckks->evaluator.rotate_vector(tmp, pow(2, i), *ckks->galois_keys, res);
    ckks->evaluator.add_inplace(res, tmp);
    tmp = res;
  }

  // Normalize res/delta to [0, 1]
  PhantomPlaintext delta;
  // PORTED: params_id() → chain_index()
  ckks->encoder.encode(0.01, res.chain_index(), res.scale(), delta);
  ckks->evaluator.multiply_plain_inplace(res, delta);
  ckks->evaluator.rescale_to_next_inplace(res);

  res = ckks->inverse(res);

  // Recover 1/res
  ckks->encoder.encode(0.01, res.chain_index(), res.scale(), delta);
  ckks->evaluator.multiply_plain_inplace(res, delta);
  ckks->evaluator.rescale_to_next_inplace(res);

  ckks->evaluator.mod_switch_to_inplace(exp_x, res.chain_index());
  ckks->evaluator.multiply(res, exp_x, res);
  ckks->evaluator.relinearize_inplace(res, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(res);
}
