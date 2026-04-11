/**
 * ckks_evaluator.cu
 *
 * Ported from vendor/nexus/cuda/src/ckks_evaluator.cu
 * API changes: params_id() → chain_index(), chain_depth() → chain_index()
 */

#include <iostream>

#include "ckks_evaluator.cuh"
#include "utils.cuh"

using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;

void CKKSEvaluator::print_decoded_pt(PhantomPlaintext &pt, int num) {
  vector<double> v;
  encoder.decode(pt, v);
  for (int i = 0; i < num; i++) {
    cout << v[i] << " ";
  }
  cout << endl;
}

void CKKSEvaluator::print_decrypted_ct(PhantomCiphertext &ct, int num) {
  PhantomPlaintext temp;
  vector<double> v;
  if (!ct.chain_index()) {
    cout << endl;
    return;
  }
  decryptor.decrypt(ct, temp);
  encoder.decode(temp, v);
  for (int i = 0; i < num; i++) {
    cout << v[i] << " ";
  }
  cout << endl;
}

vector<double> CKKSEvaluator::init_vec_with_value(double value) {
  std::vector<double> vec(encoder.slot_count(), value);
  return vec;
}

PhantomPlaintext CKKSEvaluator::init_plain_power_of_x(size_t exponent) {
  PhantomPlaintext plain_power_of_x;
  vector<double> vec = init_vec_with_value(0.0);
  vec[exponent] = 1.0;
  encoder.encode(vec, scale, plain_power_of_x);
  return plain_power_of_x;
}

PhantomCiphertext CKKSEvaluator::init_guess(PhantomCiphertext x) {
  PhantomPlaintext A, B;
  encoder.encode(-1.29054537e-04, scale, A);
  encoder.encode(1.29054537e-01, scale, B);
  return eval_line(x, A, B);
}

PhantomCiphertext CKKSEvaluator::eval_line(PhantomCiphertext x, PhantomPlaintext m, PhantomPlaintext c) {
  // PORTED: params_id() → chain_index()
  evaluator.mod_switch_to_inplace(m, x.chain_index());
  evaluator.multiply_plain_inplace(x, m);
  evaluator.rescale_to_next_inplace(x);

  evaluator.mod_switch_to_inplace(c, x.chain_index());
  x.set_scale(scale);
  evaluator.add_plain_inplace(x, c);

  return x;
}

PhantomCiphertext CKKSEvaluator::invert_sqrt(PhantomCiphertext x, int d_newt, int d_gold) {
  PhantomCiphertext res = init_guess(x);
  PhantomCiphertext y = newton_iter(x, res, d_newt);
  pair<PhantomCiphertext, PhantomCiphertext> sqrt_inv_sqrt = goldschmidt_iter(x, y, d_gold);
  return sqrt_inv_sqrt.second;
}

uint64_t CKKSEvaluator::get_modulus(PhantomCiphertext &x, int k) {
  // PORTED: params_id() → chain_index()
  const vector<phantom::arith::Modulus> &modulus = context->get_context_data(x.chain_index()).parms().coeff_modulus();
  int sz = modulus.size();
  return modulus[sz - k].value();
}

void CKKSEvaluator::re_encrypt(PhantomCiphertext &ct) {
  auto timer = Timer();
  while (ct.coeff_modulus_size() > 1) {
    evaluator.mod_switch_to_next_inplace(ct);
  }

  PhantomPlaintext temp;
  vector<double> v;
  decryptor.decrypt(ct, temp);
  encoder.decode(temp, v);
  encoder.encode(v, ct.scale(), temp);
  encryptor.encrypt(temp, ct);

  timer.stop();
  cout << timer.duration() << " milliseconds" << endl;
}

pair<PhantomCiphertext, PhantomCiphertext> CKKSEvaluator::goldschmidt_iter(PhantomCiphertext v, PhantomCiphertext y, int d) {
  PhantomCiphertext x, h, r, temp;
  PhantomPlaintext constant;

  encoder.encode(0.5, scale, constant);

  // GoldSchmidt's algorithm
  // PORTED: params_id() → chain_index()
  evaluator.mod_switch_to_inplace(v, y.chain_index());
  evaluator.multiply(v, y, x);
  evaluator.relinearize_inplace(x, *relin_keys);
  evaluator.rescale_to_next_inplace(x);

  evaluator.mod_switch_to_inplace(constant, y.chain_index());
  evaluator.multiply_plain(y, constant, h);
  evaluator.rescale_to_next_inplace(h);

  for (int i = 0; i < d; i++) {
    encoder.encode(0.5, scale, constant);

    evaluator.multiply(x, h, r);
    evaluator.relinearize_inplace(r, *relin_keys);
    evaluator.rescale_to_next_inplace(r);

    r.set_scale(scale);
    evaluator.negate(r, temp);
    evaluator.mod_switch_to_inplace(constant, temp.chain_index());
    evaluator.add_plain(temp, constant, r);

    // x = x + x*r
    evaluator.mod_switch_to_inplace(x, r.chain_index());
    evaluator.multiply(x, r, temp);
    evaluator.relinearize_inplace(temp, *relin_keys);
    evaluator.rescale_to_next_inplace(temp);

    x.set_scale(scale);
    temp.set_scale(scale);
    evaluator.mod_switch_to_inplace(x, temp.chain_index());
    evaluator.add_inplace(x, temp);

    // h = h + h*r
    evaluator.mod_switch_to_inplace(h, r.chain_index());
    evaluator.multiply(h, r, temp);
    evaluator.relinearize_inplace(temp, *relin_keys);
    evaluator.rescale_to_next_inplace(temp);

    h.set_scale(scale);
    temp.set_scale(scale);
    evaluator.mod_switch_to_inplace(h, temp.chain_index());
    evaluator.add_inplace(h, temp);
  }

  encoder.encode(2.0, scale, constant);
  evaluator.mod_switch_to_inplace(constant, h.chain_index());
  evaluator.multiply_plain_inplace(h, constant);
  evaluator.rescale_to_next_inplace(h);

  return make_pair(x, h);
}

PhantomCiphertext CKKSEvaluator::newton_iter(PhantomCiphertext x, PhantomCiphertext res, int iter) {
  for (int i = 0; i < iter; i++) {
    PhantomPlaintext three_half, neg_half;

    encoder.encode(1.5, scale, three_half);
    encoder.encode(-0.5, scale, neg_half);

    // x^2
    PhantomCiphertext res_sq;
    evaluator.square(res, res_sq);
    evaluator.relinearize_inplace(res_sq, *relin_keys);
    evaluator.rescale_to_next_inplace(res_sq);

    // -0.5*x*b
    PhantomCiphertext res_x;
    evaluator.mod_switch_to_inplace(neg_half, x.chain_index());
    evaluator.multiply_plain(x, neg_half, res_x);
    evaluator.rescale_to_next_inplace(res_x);

    // PORTED: chain_depth() → chain_index() comparison
    // Higher chain_index means deeper (more levels consumed)
    if (res.chain_index() > res_x.chain_index())
      evaluator.mod_switch_to_inplace(res_x, res.chain_index());
    else
      evaluator.mod_switch_to_inplace(res, res_x.chain_index());

    evaluator.multiply_inplace(res_x, res);
    evaluator.relinearize_inplace(res_x, *relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    // -0.5*b*x^3
    evaluator.mod_switch_to_inplace(res_sq, res_x.chain_index());
    evaluator.multiply_inplace(res_x, res_sq);
    evaluator.relinearize_inplace(res_x, *relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    // 1.5*x
    evaluator.mod_switch_to_inplace(three_half, res.chain_index());
    evaluator.multiply_plain_inplace(res, three_half);
    evaluator.rescale_to_next_inplace(res);

    // -0.5*b*x^3 + 1.5*x
    evaluator.mod_switch_to_inplace(res, res_x.chain_index());
    res_x.set_scale(scale);
    res.set_scale(scale);
    evaluator.add_inplace(res, res_x);
  }

  return res;
}

void CKKSEvaluator::eval_odd_deg9_poly(vector<double> &a, PhantomCiphertext &x, PhantomCiphertext &dest) {
  double D = x.scale();

  uint64_t p = get_modulus(x, 1);
  uint64_t q = get_modulus(x, 2);
  uint64_t r = get_modulus(x, 3);
  uint64_t s = get_modulus(x, 4);
  uint64_t t = get_modulus(x, 5);

  p = q;
  q = r;
  r = s;
  s = t;

  PhantomCiphertext x2, x3, x6;

  evaluator.square(x, x2);
  evaluator.relinearize_inplace(x2, *relin_keys);
  evaluator.rescale_to_next_inplace(x2);  // L-1

  evaluator.mod_switch_to_next_inplace(x);  // L-1
  evaluator.multiply(x2, x, x3);
  evaluator.relinearize_inplace(x3, *relin_keys);
  evaluator.rescale_to_next_inplace(x3);  // L-2

  evaluator.square(x3, x6);
  evaluator.relinearize_inplace(x6, *relin_keys);
  evaluator.rescale_to_next_inplace(x6);  // L-3

  PhantomPlaintext a1, a3, a5, a7, a9;

  // Build T1
  PhantomCiphertext T1;
  double a5_scale = D / x2.scale() * p / x3.scale() * q;
  // PORTED: params_id() → chain_index()
  encoder.encode(a[5], x2.chain_index(), a5_scale, a5);  // L-1
  evaluator.multiply_plain(x2, a5, T1);
  evaluator.rescale_to_next_inplace(T1);  // L-2

  encoder.encode(a[3], T1.chain_index(), T1.scale(), a3);  // L-2

  evaluator.add_plain_inplace(T1, a3);  // L-2
  evaluator.multiply_inplace(T1, x3);
  evaluator.relinearize_inplace(T1, *relin_keys);
  evaluator.rescale_to_next_inplace(T1);  // L-3

  // Build T2
  PhantomCiphertext T2;
  double a9_scale = D / x3.scale() * r / x6.scale() * q;
  encoder.encode(a[9], x3.chain_index(), a9_scale, a9);  // L-2
  evaluator.multiply_plain(x3, a9, T2);
  evaluator.rescale_to_next_inplace(T2);  // L-3

  PhantomCiphertext a7x;
  double a7_scale = T2.scale() / x.scale() * p;
  encoder.encode(a[7], x.chain_index(), a7_scale, a7);  // L-1 (x was modswitched)
  evaluator.multiply_plain(x, a7, a7x);
  evaluator.rescale_to_next_inplace(a7x);                       // L-2
  evaluator.mod_switch_to_inplace(a7x, T2.chain_index());       // L-3

  double mid_scale = (T2.scale() + a7x.scale()) / 2;
  T2.set_scale(mid_scale); a7x.set_scale(mid_scale);
  evaluator.add_inplace(T2, a7x);        // L-3
  evaluator.multiply_inplace(T2, x6);
  evaluator.relinearize_inplace(T2, *relin_keys);
  evaluator.rescale_to_next_inplace(T2);  // L-4

  // Build T3
  PhantomCiphertext T3;
  encoder.encode(a[1], x.chain_index(), (double)p, a1);  // L-1 (x was modswitched)
  evaluator.multiply_plain(x, a1, T3);
  evaluator.rescale_to_next_inplace(T3);  // L-2

  // T1, T2 and T3 should be on the same scale
  double mid3_scale = (T1.scale() + T2.scale() + T3.scale()) / 3;
  T1.set_scale(mid3_scale); T2.set_scale(mid3_scale); T3.set_scale(mid3_scale);

  dest = T2;
  evaluator.mod_switch_to_inplace(T1, dest.chain_index());  // L-4
  evaluator.add_inplace(dest, T1);
  evaluator.mod_switch_to_inplace(T3, dest.chain_index());  // L-4
  evaluator.add_inplace(dest, T3);
}

PhantomCiphertext CKKSEvaluator::sgn_eval(PhantomCiphertext x, int d_g, int d_f, double sgn_factor) {
  vector<double> f4_coeffs = F4_COEFFS;
  vector<double> g4_coeffs = G4_COEFFS;
  vector<double> f4_coeffs_last(10, 0.0);
  vector<double> g4_coeffs_last(10, 0.0);

  for (int i = 0; i <= 9; i++) {
    f4_coeffs[i] /= F4_SCALE;
    f4_coeffs_last[i] = f4_coeffs[i] * sgn_factor;

    g4_coeffs[i] /= G4_SCALE;
    g4_coeffs_last[i] = g4_coeffs[i] * sgn_factor;
  }

  PhantomCiphertext dest = x;

  for (int i = 0; i < d_g; i++) {
    if (i == d_g - 1) {
      eval_odd_deg9_poly(g4_coeffs_last, dest, dest);
    } else {
      eval_odd_deg9_poly(g4_coeffs, dest, dest);
    }
  }

  for (int i = 0; i < d_f; i++) {
    if (i == d_f - 1) {
      eval_odd_deg9_poly(f4_coeffs_last, dest, dest);
    } else {
      eval_odd_deg9_poly(f4_coeffs, dest, dest);
    }
  }

  return dest;
}

double CKKSEvaluator::calculate_MAE(vector<double> &y_true, PhantomCiphertext &ct, int N) {
  PhantomPlaintext temp;
  vector<double> y_pred;

  decryptor.decrypt(ct, temp);
  encoder.decode(temp, y_pred);

  double sum_absolute_errors = 0.0;
  for (size_t i = 0; i < (size_t)N; ++i) {
    sum_absolute_errors += abs(y_true[i] - y_pred[i]);
  }

  return sum_absolute_errors / N;
}

PhantomCiphertext CKKSEvaluator::exp(PhantomCiphertext x) {
  PhantomPlaintext one, inverse_128;

  // PORTED: params_id() → chain_index()
  encoder.encode(0.0078125, x.chain_index(), x.scale(), inverse_128);
  evaluator.multiply_plain_inplace(x, inverse_128);
  evaluator.rescale_to_next_inplace(x);

  encoder.encode(1.0, x.chain_index(), x.scale(), one);
  evaluator.add_plain_inplace(x, one);

  // x^128
  for (int i = 0; i < (int)log2(128); i++) {
    evaluator.square(x, x);
    evaluator.relinearize_inplace(x, *relin_keys);
    evaluator.rescale_to_next_inplace(x);
  }

  return x;
}

PhantomCiphertext CKKSEvaluator::inverse(PhantomCiphertext x, int iter) {
  PhantomCiphertext y, tmp, res;
  PhantomPlaintext one;

  // PORTED: params_id() → chain_index()
  encoder.encode(1.0, x.chain_index(), x.scale(), one);
  evaluator.sub_plain(x, one, y);
  evaluator.negate_inplace(y);
  evaluator.add_plain(y, one, tmp);

  res = tmp;

  for (int i = 0; i < iter; i++) {
    evaluator.square_inplace(y);
    evaluator.relinearize_inplace(y, *relin_keys);
    evaluator.rescale_to_next_inplace(y);

    encoder.encode(1.0, y.chain_index(), y.scale(), one);
    evaluator.add_plain(y, one, tmp);

    evaluator.mod_switch_to_inplace(res, tmp.chain_index());
    evaluator.multiply_inplace(res, tmp);
    evaluator.relinearize_inplace(res, *relin_keys);
    evaluator.rescale_to_next_inplace(res);
  }

  return res;
}

void nexus::Evaluator::add_inplace_reduced_error(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) {
  size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
  size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();

  if (encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size) {
    encrypted1.set_scale(encrypted2.scale());
    add_inplace(encrypted1, encrypted2);
    return;
  } else if (encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size) {
    // PORTED: params_id() → chain_index()
    auto &context_data = context->get_context_data(encrypted2.chain_index());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted2_adjusted;

    double scale_adjust = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()) / (encrypted2.scale() * encrypted2.scale());
    multiply_const(encrypted2, scale_adjust, encrypted2_adjusted);
    encrypted2_adjusted.set_scale(encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()));
    rescale_to_next_inplace(encrypted2_adjusted);
    mod_switch_to_inplace(encrypted2_adjusted, encrypted1.chain_index());
    encrypted1.set_scale(encrypted2_adjusted.scale());
    add_inplace(encrypted1, encrypted2_adjusted);
  } else {
    auto &context_data = context->get_context_data(encrypted1.chain_index());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted1_adjusted;

    double scale_adjust = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()) / (encrypted1.scale() * encrypted1.scale());
    multiply_const(encrypted1, scale_adjust, encrypted1_adjusted);
    encrypted1_adjusted.set_scale(encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()));
    rescale_to_next_inplace(encrypted1_adjusted);
    mod_switch_to_inplace(encrypted1_adjusted, encrypted2.chain_index());
    encrypted1_adjusted.set_scale(encrypted2.scale());
    add(encrypted1_adjusted, encrypted2, encrypted1);
  }
}

void nexus::Evaluator::sub_inplace_reduced_error(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) {
  size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
  size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();

  if (encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size) {
    encrypted1.set_scale(encrypted2.scale());
    sub_inplace(encrypted1, encrypted2);
    return;
  } else if (encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size) {
    auto &context_data = context->get_context_data(encrypted2.chain_index());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted2_adjusted;

    double scale_adjust = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()) / (encrypted2.scale() * encrypted2.scale());
    multiply_const(encrypted2, scale_adjust, encrypted2_adjusted);
    encrypted2_adjusted.set_scale(encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()));
    rescale_to_next_inplace(encrypted2_adjusted);
    mod_switch_to_inplace(encrypted2_adjusted, encrypted1.chain_index());
    encrypted1.set_scale(encrypted2_adjusted.scale());
    sub_inplace(encrypted1, encrypted2_adjusted);
  } else {
    auto &context_data = context->get_context_data(encrypted1.chain_index());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted1_adjusted;

    double scale_adjust = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()) / (encrypted1.scale() * encrypted1.scale());
    multiply_const(encrypted1, scale_adjust, encrypted1_adjusted);
    encrypted1_adjusted.set_scale(encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()));
    rescale_to_next_inplace(encrypted1_adjusted);
    mod_switch_to_inplace(encrypted1_adjusted, encrypted2.chain_index());
    encrypted1_adjusted.set_scale(encrypted2.scale());
    sub(encrypted1_adjusted, encrypted2, encrypted1);
  }
}

void nexus::Evaluator::multiply_inplace_reduced_error(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys) {
  size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
  size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();

  if (encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size) {
    encrypted1.set_scale(encrypted2.scale());
    multiply_inplace(encrypted1, encrypted2);
    relinearize_inplace(encrypted1, relin_keys);
    return;
  } else if (encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size) {
    auto &context_data = context->get_context_data(encrypted2.chain_index());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted2_adjusted;

    double scale_adjust = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()) / (encrypted2.scale() * encrypted2.scale());
    multiply_const(encrypted2, scale_adjust, encrypted2_adjusted);
    encrypted2_adjusted.set_scale(encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()));
    rescale_to_next_inplace(encrypted2_adjusted);
    mod_switch_to_inplace(encrypted2_adjusted, encrypted1.chain_index());
    encrypted1.set_scale(encrypted2_adjusted.scale());
    multiply_inplace(encrypted1, encrypted2_adjusted);
  } else {
    auto &context_data = context->get_context_data(encrypted1.chain_index());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted1_adjusted;

    double scale_adjust = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()) / (encrypted1.scale() * encrypted1.scale());
    multiply_const(encrypted1, scale_adjust, encrypted1_adjusted);
    encrypted1_adjusted.set_scale(encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()));
    rescale_to_next_inplace(encrypted1_adjusted);
    mod_switch_to_inplace(encrypted1_adjusted, encrypted2.chain_index());
    encrypted1_adjusted.set_scale(encrypted2.scale());
    multiply(encrypted1_adjusted, encrypted2, encrypted1);
  }

  relinearize_inplace(encrypted1, relin_keys);
}
