#pragma once

#include "ckks_evaluator.cuh"

namespace nexus {
using namespace std;
using namespace phantom;

class LNEvaluator {
 private:
  CKKSEvaluator *ckks = nullptr;

 public:
  LNEvaluator(CKKSEvaluator &ckks) : ckks(&ckks) {}
  void layer_norm(PhantomCiphertext &x, PhantomCiphertext &res, int len);
};
}  // namespace nexus
