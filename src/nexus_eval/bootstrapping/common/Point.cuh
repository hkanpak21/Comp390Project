#pragma once
#include <NTL/RR.h>

#include <iostream>

using namespace std;


class Point {
 public:
  NTL::RR x;
  NTL::RR y;

  long locmm;

  Point() = default;
  Point(NTL::RR _x, NTL::RR _y);
};
