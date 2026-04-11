/**
 * ckks_evaluator.cuh
 *
 * Ported from vendor/nexus/cuda/src/ckks_evaluator.cuh
 * Uses NEXUS Phantom fork (vendor/phantom/) with save/load backported.
 */

#pragma once

#include <complex>
#include <vector>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"
#include "plaintext.h"

namespace nexus {
using namespace std;
using namespace phantom;

class Encoder {
 private:
  PhantomContext *context;
  PhantomCKKSEncoder *encoder;

 public:
  Encoder() = default;

  Encoder(PhantomContext *context, PhantomCKKSEncoder *encoder) {
    this->context = context;
    this->encoder = encoder;
  }

  inline size_t slot_count() { return encoder->slot_count(); }

  inline void reset_sparse_slots() { encoder->reset_sparse_slots(); }

  // Vector (of doubles or complexes) inputs — with chain_index
  inline void encode(vector<double> values, size_t chain_index, double scale, PhantomPlaintext &plain) {
    if (values.size() == 1) {
      encode(values[0], chain_index, scale, plain);
      return;
    }
    values.resize(encoder->slot_count(), 0.0);
    encoder->encode(*context, values, scale, plain, chain_index);
  }

  // Vector encode — default chain_index
  inline void encode(vector<double> values, double scale, PhantomPlaintext &plain) {
    if (values.size() == 1) {
      encode(values[0], scale, plain);
      return;
    }
    values.resize(encoder->slot_count(), 0.0);
    encoder->encode(*context, values, scale, plain);
  }

  inline void encode(vector<complex<double>> complex_values, double scale, PhantomPlaintext &plain) {
    // Convert std::complex<double> to cuDoubleComplex for Phantom API
    vector<cuDoubleComplex> cu_values(encoder->slot_count());
    for (size_t i = 0; i < complex_values.size() && i < cu_values.size(); i++) {
      cu_values[i] = make_cuDoubleComplex(complex_values[i].real(), complex_values[i].imag());
    }
    for (size_t i = complex_values.size(); i < cu_values.size(); i++) {
      cu_values[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    encoder->encode(*context, cu_values, scale, plain);
  }

  // Scalar value inputs (fill all slots)
  inline void encode(double value, size_t chain_index, double scale, PhantomPlaintext &plain) {
    vector<double> values(encoder->slot_count(), value);
    encoder->encode(*context, values, scale, plain, chain_index);
  }

  inline void encode(double value, double scale, PhantomPlaintext &plain) {
    vector<double> values(encoder->slot_count(), value);
    encoder->encode(*context, values, scale, plain);
  }

  inline void encode(complex<double> complex_value, double scale, PhantomPlaintext &plain) {
    vector<cuDoubleComplex> cu_values(encoder->slot_count(),
        make_cuDoubleComplex(complex_value.real(), complex_value.imag()));
    encoder->encode(*context, cu_values, scale, plain);
  }

  template <typename T, typename = std::enable_if_t<std::is_same<std::remove_cv_t<T>, double>::value || std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
  inline void decode(PhantomPlaintext &plain, vector<T> &values) {
    encoder->decode(*context, plain, values);
  }
};

class Encryptor {
 private:
  PhantomContext *context;
  PhantomPublicKey *public_key;

 public:
  Encryptor() = default;

  Encryptor(PhantomContext *context, PhantomPublicKey *public_key) {
    this->context = context;
    this->public_key = public_key;
  }

  inline void encrypt(PhantomPlaintext &plain, PhantomCiphertext &ct) {
    public_key->encrypt_asymmetric(*context, plain, ct);
  }
};

class Evaluator {
 private:
  PhantomContext *context;
  PhantomCKKSEncoder *encoder;

 public:
  Evaluator() = default;
  Evaluator(PhantomContext *context, PhantomCKKSEncoder *encoder) {
    this->context = context;
    this->encoder = encoder;
  }

  // Mod switch
  inline void mod_switch_to_next_inplace(PhantomCiphertext &ct) {
    ::mod_switch_to_next_inplace(*context, ct);
  }

  inline void mod_switch_to_inplace(PhantomCiphertext &ct, size_t chain_index) {
    ::mod_switch_to_inplace(*context, ct, chain_index);
  }

  inline void mod_switch_to_inplace(PhantomPlaintext &pt, size_t chain_index) {
    ::mod_switch_to_inplace(*context, pt, chain_index);
  }

  inline void rescale_to_next_inplace(PhantomCiphertext &ct) {
    ::rescale_to_next_inplace(*context, ct);
  }

  // Relinearization
  inline void relinearize_inplace(PhantomCiphertext &ct, const PhantomRelinKey &relin_keys) {
    ::relinearize_inplace(*context, ct, relin_keys);
  }

  // Multiplication
  inline void square(PhantomCiphertext &ct, PhantomCiphertext &dest) {
    multiply(ct, ct, dest);
  }

  inline void square_inplace(PhantomCiphertext &ct) {
    multiply_inplace(ct, ct);
  }

  inline void multiply(PhantomCiphertext &ct1, const PhantomCiphertext &ct2, PhantomCiphertext &dest) {
    if (&ct2 == &dest) {
      multiply_inplace(dest, ct1);
    } else {
      dest = ct1;
      multiply_inplace(dest, ct2);
    }
  }

  inline void multiply_inplace(PhantomCiphertext &ct1, const PhantomCiphertext &ct2) {
    if (ct1.scale() != ct2.scale()) {
      ct1.set_scale(ct2.scale());
    }
    ::multiply_inplace(*context, ct1, ct2);
  }

  inline void multiply_plain(PhantomCiphertext &ct, PhantomPlaintext &plain, PhantomCiphertext &dest) {
    dest = ::multiply_plain(*context, ct, plain);
  }

  inline void multiply_plain_inplace(PhantomCiphertext &ct, PhantomPlaintext &plain) {
    if (ct.scale() != plain.scale()) {
      const_cast<double&>(plain.scale()) = ct.scale();
    }
    ::multiply_plain_inplace(*context, ct, plain);
  }

  // Addition
  inline void add_plain(const PhantomCiphertext &ct, PhantomPlaintext &plain, PhantomCiphertext &dest) {
    dest = ::add_plain(*context, ct, plain);
  }

  inline void add_plain_inplace(PhantomCiphertext &ct, PhantomPlaintext &plain) {
    // Force scale match for plaintext operations
    if (ct.scale() != plain.scale()) {
      const_cast<double&>(plain.scale()) = ct.scale();
    }
    ::add_plain_inplace(*context, ct, plain);
  }

  inline void add(PhantomCiphertext &ct1, const PhantomCiphertext &ct2, PhantomCiphertext &dest) {
    if (ct1.scale() != ct2.scale()) {
      ct1.set_scale(ct2.scale());
    }
    dest = ::add(*context, ct1, ct2);
  }

  inline void add_inplace(PhantomCiphertext &ct1, const PhantomCiphertext &ct2) {
    // Force scale match — NEXUS code relies on manual scale normalization
    // Our Phantom is stricter about scale matching than the NEXUS fork
    if (ct1.scale() != ct2.scale()) {
      ct1.set_scale(ct2.scale());
    }
    ::add_inplace(*context, ct1, ct2);
  }

  inline void add_many(vector<PhantomCiphertext> &cts, PhantomCiphertext &dest) {
    size_t size = cts.size();
    if (size < 2) throw invalid_argument("add_many requires at least 2 ciphertexts");
    add(cts[0], cts[1], dest);
    for (size_t i = 2; i < size; i++) {
      add_inplace(dest, cts[i]);
    }
  }

  // Subtraction
  inline void sub_plain(PhantomCiphertext &ct, PhantomPlaintext &plain, PhantomCiphertext &dest) {
    dest = ct;
    sub_plain_inplace(dest, plain);
  }

  inline void sub_plain_inplace(PhantomCiphertext &ct, PhantomPlaintext &plain) {
    if (ct.scale() != plain.scale()) {
      const_cast<double&>(plain.scale()) = ct.scale();
    }
    ::sub_plain_inplace(*context, ct, plain);
  }

  inline void sub(PhantomCiphertext &ct1, const PhantomCiphertext &ct2, PhantomCiphertext &dest) {
    if (&ct2 == &dest) {
      sub_inplace(dest, ct1);
      negate_inplace(dest);
    } else {
      dest = ct1;
      sub_inplace(dest, ct2);
    }
  }

  inline void sub_inplace(PhantomCiphertext &ct1, const PhantomCiphertext &ct2) {
    if (ct1.scale() != ct2.scale()) {
      ct1.set_scale(ct2.scale());
    }
    ::sub_inplace(*context, ct1, ct2);
  }

  // Rotation
  inline void rotate_vector(PhantomCiphertext &ct, int steps, PhantomGaloisKey &galois_keys, PhantomCiphertext &dest) {
    dest = ::rotate_vector(*context, ct, steps, galois_keys);
  }

  inline void rotate_vector_inplace(PhantomCiphertext &ct, int steps, PhantomGaloisKey &galois_keys) {
    ::rotate_vector_inplace(*context, ct, steps, galois_keys);
  }

  // Negation
  inline void negate(PhantomCiphertext &ct, PhantomCiphertext &dest) {
    dest = ct;
    negate_inplace(dest);
  }

  inline void negate_inplace(PhantomCiphertext &ct) {
    ::negate_inplace(*context, ct);
  }

  // Galois
  inline void apply_galois(PhantomCiphertext &ct, uint32_t elt, PhantomGaloisKey &galois_keys, PhantomCiphertext &dest) {
    dest = ::apply_galois(*context, ct, elt, galois_keys);
  }

  inline void apply_galois_inplace(PhantomCiphertext &ct, int step, PhantomGaloisKey &galois_keys) {
    auto N = context->get_context_data(0).parms().poly_modulus_degree();
    uint32_t m = 2 * (uint32_t)N;
    uint32_t elt;
    if (step == 0) {
      elt = m - 1; // conjugation
    } else {
      uint32_t gen = 5;
      int abs_step = step < 0 ? -step : step;
      elt = 1;
      for (int i = 0; i < abs_step; i++) elt = (elt * gen) % m;
      if (step < 0) elt = (m + 1 - elt) % m; // inverse
    }
    ::apply_galois_inplace(*context, ct, elt, galois_keys);
  }

  // Complex Conjugate — needed by bootstrapping (slottocoeff_full_3, etc.)
  inline void complex_conjugate(PhantomCiphertext &ct, const PhantomGaloisKey &galois_keys, PhantomCiphertext &dest) {
    dest = ct;
    complex_conjugate_inplace(dest, galois_keys);
  }

  inline void complex_conjugate_inplace(PhantomCiphertext &ct, const PhantomGaloisKey &galois_keys) {
    ::complex_conjugate_inplace(*context, ct, galois_keys);
  }

  // NTT transforms
  inline void transform_from_ntt(const PhantomCiphertext &ct, PhantomCiphertext &dest) {
    dest = ct;
    transform_from_ntt_inplace(dest);
  }

  inline void transform_from_ntt_inplace(PhantomCiphertext &ct) {
    auto rns_coeff_count = ct.poly_modulus_degree() * ct.coeff_modulus_size();
    const auto stream = ct.data_ptr().get_stream();
    for (size_t i = 0; i < ct.size(); i++) {
      uint64_t *ci = ct.data() + i * rns_coeff_count;
      nwt_2d_radix8_backward_inplace(ci, context->gpu_rns_tables(), ct.coeff_modulus_size(), 0, stream);
    }
    ct.set_ntt_form(false);
  }

  inline void transform_to_ntt(const PhantomCiphertext &ct, PhantomCiphertext &dest) {
    dest = ct;
    transform_to_ntt_inplace(dest);
  }

  inline void transform_to_ntt_inplace(PhantomCiphertext &ct) {
    auto rns_coeff_count = ct.poly_modulus_degree() * ct.coeff_modulus_size();
    const auto stream = ct.data_ptr().get_stream();
    for (size_t i = 0; i < ct.size(); i++) {
      uint64_t *ci = ct.data() + i * rns_coeff_count;
      nwt_2d_radix8_forward_inplace(ci, context->gpu_rns_tables(), ct.coeff_modulus_size(), 0, stream);
    }
    ct.set_ntt_form(true);
  }

  // Const operations
  inline void multiply_const(const PhantomCiphertext &ct, double value, PhantomCiphertext &dest) {
    dest = ct;
    multiply_const_inplace(dest, value);
  }

  inline void multiply_const_inplace(PhantomCiphertext &ct, double value) {
    PhantomPlaintext const_plain;
    vector<double> values(encoder->slot_count(), value);
    encoder->encode(*context, values, ct.scale(), const_plain);
    // PORTED: mod_switch_to_inplace with chain_index instead of parms_id
    mod_switch_to_inplace(const_plain, ct.chain_index());
    multiply_plain_inplace(ct, const_plain);
  }

  inline void add_const(PhantomCiphertext &ct, double value, PhantomCiphertext &dest) {
    dest = ct;
    add_const_inplace(dest, value);
  }

  inline void add_const_inplace(PhantomCiphertext &ct, double value) {
    PhantomPlaintext const_plain;
    vector<double> values(encoder->slot_count(), value);
    encoder->encode(*context, values, ct.scale(), const_plain);
    mod_switch_to_inplace(const_plain, ct.chain_index());
    add_plain_inplace(ct, const_plain);
  }

  inline void add_reduced_error(const PhantomCiphertext &ct1, const PhantomCiphertext &ct2, PhantomCiphertext &dest) {
    if (&ct2 == &dest) {
      add_inplace_reduced_error(dest, ct1);
    } else {
      dest = ct1;
      add_inplace_reduced_error(dest, ct2);
    }
  }

  void add_inplace_reduced_error(PhantomCiphertext &ct1, const PhantomCiphertext &ct2);

  inline void sub_reduced_error(const PhantomCiphertext &ct1, const PhantomCiphertext &ct2, PhantomCiphertext &dest) {
    dest = ct1;
    sub_inplace_reduced_error(dest, ct2);
  }

  void sub_inplace_reduced_error(PhantomCiphertext &ct1, const PhantomCiphertext &ct2);

  inline void multiply_reduced_error(const PhantomCiphertext &ct1, const PhantomCiphertext &ct2, const PhantomRelinKey &relin_keys, PhantomCiphertext &dest) {
    if (&ct2 == &dest) {
      multiply_inplace_reduced_error(dest, ct1, relin_keys);
    } else {
      dest = ct1;
      multiply_inplace_reduced_error(dest, ct2, relin_keys);
    }
  }

  void multiply_inplace_reduced_error(PhantomCiphertext &ct1, const PhantomCiphertext &ct2, const PhantomRelinKey &relin_keys);

  inline void double_inplace(PhantomCiphertext &ct) const {
    ::add_inplace(*context, ct, ct);
  }

  template <typename T, typename = std::enable_if_t<std::is_same<std::remove_cv_t<T>, double>::value || std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
  inline void multiply_vector_reduced_error(PhantomCiphertext &ct, std::vector<T> &values, PhantomCiphertext &dest) {
    dest = ct;
    multiply_vector_inplace_reduced_error(dest, values);
  }

  inline void multiply_vector_inplace_reduced_error(PhantomCiphertext &ct, vector<double> &values) {
    PhantomPlaintext plain;
    values.resize(encoder->slot_count(), 0.0);
    encoder->encode(*context, values, ct.scale(), plain);
    mod_switch_to_inplace(plain, ct.chain_index());
    multiply_plain_inplace(ct, plain);
  }

  inline void multiply_vector_inplace_reduced_error(PhantomCiphertext &ct, vector<complex<double>> &values) {
    PhantomPlaintext plain;
    vector<cuDoubleComplex> cu_values(encoder->slot_count(), make_cuDoubleComplex(0.0, 0.0));
    for (size_t i = 0; i < values.size() && i < cu_values.size(); i++) {
      cu_values[i] = make_cuDoubleComplex(values[i].real(), values[i].imag());
    }
    encoder->encode(*context, cu_values, ct.scale(), plain);
    mod_switch_to_inplace(plain, ct.chain_index());
    multiply_plain_inplace(ct, plain);
  }
};

class Decryptor {
 private:
  PhantomContext *context;
  PhantomSecretKey *decryptor;

 public:
  Decryptor() = default;
  Decryptor(PhantomContext *context, PhantomSecretKey *decryptor) {
    this->context = context;
    this->decryptor = decryptor;
  }

  inline void decrypt(PhantomCiphertext &ct, PhantomPlaintext &plain) {
    decryptor->decrypt(*context, ct, plain);
  }

  // Selective Galois key generation (much less memory than all keys)
  inline void create_galois_keys_from_steps(vector<int> &steps, PhantomGaloisKey &galois_keys) {
    galois_keys = decryptor->create_galois_keys_from_steps(*context, steps);
  }

  inline void create_galois_keys_from_elts(vector<uint32_t> &elts, PhantomGaloisKey &galois_keys) {
    galois_keys = decryptor->create_galois_keys_from_elts(*context, elts);
  }
};

class CKKSEvaluator {
 private:
  // Sign function coefficients
  vector<double> F4_COEFFS = {0, 315, 0, -420, 0, 378, 0, -180, 0, 35};
  int F4_SCALE = (1 << 7);
  vector<double> G4_COEFFS = {0, 5850, 0, -34974, 0, 97015, 0, -113492, 0, 46623};
  int G4_SCALE = (1 << 10);

  // Helper functions
  uint64_t get_modulus(PhantomCiphertext &x, int k);

  PhantomCiphertext init_guess(PhantomCiphertext x);
  PhantomCiphertext eval_line(PhantomCiphertext x, PhantomPlaintext m, PhantomPlaintext c);

  // Evaluation functions
  PhantomCiphertext newton_iter(PhantomCiphertext x, PhantomCiphertext res, int iter);
  pair<PhantomCiphertext, PhantomCiphertext> goldschmidt_iter(PhantomCiphertext v, PhantomCiphertext y, int d = 1);
  void eval_odd_deg9_poly(vector<double> &a, PhantomCiphertext &x, PhantomCiphertext &dest);

 public:
  // Memory managed outside of the evaluator
  PhantomContext *context;
  PhantomRelinKey *relin_keys;
  PhantomGaloisKey *galois_keys;
  std::vector<std::uint32_t> galois_elts;

  // Component classes
  Encoder encoder;
  Encryptor encryptor;
  Evaluator evaluator;
  Decryptor decryptor;

  size_t degree;
  double scale;
  size_t slot_count;

  CKKSEvaluator(PhantomContext *context, PhantomPublicKey *pk, PhantomSecretKey *sk,
                PhantomCKKSEncoder *enc, PhantomRelinKey *relin_keys, PhantomGaloisKey *galois_keys,
                double scale, vector<uint32_t> galois_elts = {}) {
    this->context = context;
    this->relin_keys = relin_keys;
    this->galois_keys = galois_keys;
    this->galois_elts = galois_elts;

    this->scale = scale;
    this->slot_count = enc->slot_count();
    this->degree = this->slot_count * 2;

    // Instantiate the component classes
    Encoder ckks_encoder(context, enc);
    this->encoder = ckks_encoder;

    Encryptor ckks_encryptor(context, pk);
    this->encryptor = ckks_encryptor;

    Evaluator ckks_evaluator(context, enc);
    this->evaluator = ckks_evaluator;

    Decryptor ckks_decryptor(context, sk);
    this->decryptor = ckks_decryptor;
  }

  // Helper functions
  vector<double> init_vec_with_value(double value);
  PhantomPlaintext init_plain_power_of_x(size_t exponent);

  void re_encrypt(PhantomCiphertext &ct);
  void print_decrypted_ct(PhantomCiphertext &ct, int num);
  void print_decoded_pt(PhantomPlaintext &pt, int num);

  // Evaluation functions
  PhantomCiphertext sgn_eval(PhantomCiphertext x, int d_g, int d_f, double sgn_factor = 0.5);
  PhantomCiphertext invert_sqrt(PhantomCiphertext x, int d_newt = 20, int d_gold = 1);
  PhantomCiphertext exp(PhantomCiphertext x);
  PhantomCiphertext inverse(PhantomCiphertext x, int iter = 4);

  // Metrics calculation functions
  double calculate_MAE(vector<double> &y_true, PhantomCiphertext &ct, int N);
};

}  // namespace nexus
