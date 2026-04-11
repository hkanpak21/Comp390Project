#include "Point.cuh"
using namespace NTL;
// Shadow NTL::min/max with CUDA versions to prevent conflict
template<typename T> static inline T min(T a, T b) { return a < b ? a : b; }
template<typename T> static inline T max(T a, T b) { return a > b ? a : b; }

Point::Point(RR _x, RR _y) {
  x = _x;
  y = _y;
}
