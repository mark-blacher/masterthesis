#include "avx2sort.h"
#include <functional>
#include <random>
#include <vector>

using namespace std;

/* this code demonstrates how to use avx2sort.h for sorting integers */

int main() {
  int n = 1000000;

  /* create a vector with random integers */
  vector<int> a(n);
  auto rand_int = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
  for (int i = 0; i < n; ++i) {
    a[i] = rand_int();
  }
  auto b = a;
  auto c = a;

  /* single-threaded sort */
  avx2::quicksort(a.data(), n);

  /* multi-threaded sort if OpenMP enabled */
  avx2::quicksort_omp(b.data(), n);

  /* instead of nth_element(begin(c), begin(c) + (n / 2), end(c)) */
  avx2::quickselect(c.data(), n, n / 2);

  puts("done");
}
