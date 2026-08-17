#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace arm_ai {

struct SafeStats {
  static double mean(const std::vector<double> &v) noexcept {
    if (v.empty())
      return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) /
           static_cast<double>(v.size());
  }

  static double percentile(std::vector<double> v, double p) {
    if (v.empty())
      return 0.0;
    std::sort(v.begin(), v.end());

    double rank = (p / 100.0) * static_cast<double>(v.size());
    size_t idx = static_cast<size_t>(std::ceil(rank));

    if (idx > 0) {
      idx--;
    }
    idx = std::min(idx, v.size() - 1);

    return v[idx];
  }

  static double standard_deviation(const std::vector<double> &v) noexcept {
    if (v.size() < 2)
      return 0.0;
    double avg = mean(v);
    double sum_sq_diff = 0.0;
    for (double val : v) {
      double diff = val - avg;
      sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / static_cast<double>(v.size() - 1));
  }
};

} // namespace arm_ai
