#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "ckks_evaluator.cuh"

namespace nexus {
using namespace std;
using namespace phantom;

class MMEvaluator {
 private:
  CKKSEvaluator *ckks = nullptr;

  void enc_compress_ciphertext(vector<double> &values, PhantomCiphertext &ct);
  vector<PhantomCiphertext> decompress_ciphertext(PhantomCiphertext &encrypted);

 public:
  MMEvaluator(CKKSEvaluator &ckks) : ckks(&ckks) {}

  // Helper functions
  inline vector<vector<double>> read_matrix(const std::string &filename, int rows, int cols) {
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Cannot open file: " << filename << std::endl;
      return matrix;
    }
    std::string line;
    for (int i = 0; i < rows; ++i) {
      if (std::getline(file, line)) {
        std::istringstream iss(line);
        for (int j = 0; j < cols; ++j) {
          if (!(iss >> matrix[i][j])) {
            std::cerr << "Read error: " << filename << " (row: " << i << ", col: " << j << ")" << std::endl;
          }
        }
      }
    }
    file.close();
    return matrix;
  }

  inline vector<vector<double>> transpose_matrix(const vector<vector<double>> &matrix) {
    if (matrix.empty()) return {};
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        transposed[j][i] = matrix[i][j];
      }
    }
    return transposed;
  }

  // Evaluation function (original N=8192 compress/decompress path)
  void matrix_mul(vector<vector<double>> &x, vector<vector<double>> &y, vector<PhantomCiphertext> &res);
  void multiply_power_of_x(PhantomCiphertext &encrypted, PhantomCiphertext &destination, int index);

  // Unified N=65536 MatMul — no compress/decompress needed.
  // Input: x_cts[i] = encrypted row vector (768 values in 32768 slots)
  //        weights[j][k] = plaintext weight (inner_dim rows × slot values)
  // Output: res[col] = sum_j(x_cts[col] * weights[j]) for each output column
  // Consumes 2 levels (multiply_plain + rescale).
  void matrix_mul_unified(
      vector<PhantomCiphertext> &x_cts,
      vector<vector<double>> &weights,
      int n_columns,
      vector<PhantomCiphertext> &res);
};
}  // namespace nexus
