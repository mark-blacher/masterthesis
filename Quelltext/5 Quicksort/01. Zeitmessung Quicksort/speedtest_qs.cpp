#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "avx2sort.h"

using namespace std;
using namespace std::chrono;

#if defined(__INTEL_COMPILER)
#include <ipp.h>
#endif

void test_correctness(){
  for (int k = 0; k < 8000; ++k) {
    volatile int n = k;
    vector<int> v(k);
    auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
    for (int i = 0; i < n; ++i) {
      v[i] = die();
    }
    auto a = v;
    auto b = v;
    sort(begin(a), end(a));
    avx2::quicksort(b.data(), k);
    if(a != b){
      cerr << "fail " << n << endl;
      exit(-1);
    }
  }
  cout << "quicksort sortiert korrekt!" << endl;
}

int main() {
  test_correctness();
  bool calculate_speedups_for_diagram = false; /* auf true setzen falls Zeitmessung für Diagramm */
#if defined(__INTEL_COMPILER)
  calculate_speedups_for_diagram = true;
#endif
  if(calculate_speedups_for_diagram){
    volatile int n = 0;

    vector<int> vec_n{10000, 100000, 1000000, 10000000, 100000000, 1000000000};
    vector<int> vec_runs{8000, 1000, 200, 40, 10, 1};
    vector<double> vec_speedups(6 * 2, 0);

    int repeating = 10;
    for (int s = 0; s < repeating; ++s) {
      for (int i = 0; i < 6; ++i) {
        n = vec_n[i];
        int runs = vec_runs[i];

        vector<int> v(static_cast<unsigned long long int>(n));
        auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
//        auto die = bind(uniform_int_distribution<int>{1000000, 1000100}, default_random_engine{std::random_device{}()});
        for (int r = 0; r < n; ++r) {
          v[r] = die();
        }

        vector<int> a = v;

        double copy_time = 0;
        {
          auto tic = system_clock::now();
          for (int j = 0; j < runs; ++j) {
            memcpy(a.data(), v.data(), sizeof(int) * n);
          }
          auto toc = system_clock::now();
        }

        double dur_std = 0;
        {
          auto tic = system_clock::now();
          for (int j = 0; j < runs; ++j) {
            memcpy(a.data(), v.data(), sizeof(int) * n);
            sort(begin(a), end(a));
          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          dur_std = dur;
          if(!is_sorted(begin(a), end(a))){cerr << "std failed!!!"; exit(-1);} ;
        }
        {
          /* quicksort vectorized */
          auto tic = system_clock::now();
          for (int j = 0; j < runs; ++j) {
            memcpy(a.data(), v.data(), sizeof(int) * n);
            avx2::quicksort(a.data(), n);          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          vec_speedups[i] += dur > 0.000001 ? dur_std/double(dur) : 0;
          if(!is_sorted(begin(a), end(a))){cerr << "qs failed!!!"; exit(-1);} ;
        }

        {
#if defined(__INTEL_COMPILER)
          auto tic = system_clock::now();

          IppSizeL sz;
          Ipp8u *buffer;
          ippsSortRadixGetBufferSize_L(n, ipp32s, &sz);
          buffer = (Ipp8u *)malloc(sz);

          for (int j = 0; j < runs; ++j) {
            memcpy(a.data(), v.data(), sizeof(int) * n);
            ippsSortRadixAscend_32s_I_L(a.data(), n, buffer);
          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          vec_speedups[i + 6] += dur > 0.00001 ? dur_std / double(dur) : 0;
          if(!is_sorted(begin(a), end(a))){cerr << "radix failed!!!"; exit(-1);} ;
#endif
        }
        cout << "n: " << n << " in repeating " << s << " finished" << endl;
      }
      cout << endl;
    }
    cout << "data_from_cpp = c(";
    for (int k = 0; k < 6 * 2 - 1; ++k) {
      cout << (vec_speedups[k] / repeating) << ", ";
    }
    cout << (vec_speedups[6 * 2 - 1] / repeating) << ")\n";
  }else {
    volatile int n = 1000000;  // size of vector
    const int runs = 100; // repetitions of calculation
    vector<int> v(static_cast<unsigned long long int>(n));

    auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
//    auto die = bind(uniform_int_distribution<int>{1000000, 1000010}, default_random_engine{std::random_device{}()});

    for (int i = 0; i < n; ++i) {
      v[i] = die();
    }

    vector<int> a = v;
    vector<int> b = v;
    vector<int> d = v;

    double copy_time = 0;
    {
      auto tic = system_clock::now();
      for (int j = 0; j < runs; ++j) {
        memcpy(a.data(), v.data(), sizeof(int) * n);
      }
      auto toc = system_clock::now();
      copy_time = duration<double>(toc - tic).count();
    }

    double dur_std = 0;
    {
      cout << "std::sort: ";
      auto tic = system_clock::now();
      for (int j = 0; j < runs; ++j) {
        memcpy(a.data(), v.data(), sizeof(int) * n);
        sort(begin(a), end(a));
      }
      auto toc = system_clock::now();
      auto dur = duration<double>(toc - tic).count() - copy_time;
      dur_std = dur;
      cout << "\n" << dur << " sec\n\n";
    }

    {
      cout << "quicksort vectorized:";
      auto tic = system_clock::now();
      for (int j = 0; j < runs; ++j) {
        memcpy(b.data(), v.data(), sizeof(int) * n);
        avx2::quicksort(b.data(), n);
      }
      auto toc = system_clock::now();
      auto dur = duration<double>(toc - tic).count() - copy_time;
      cout << "\n" << dur << " sec";
      if (dur_std > 0.001) cout << "\nspeed-up: " << dur_std / double(dur);
      cout << boolalpha << "\nis_sorted: " << is_sorted(begin(b), end(b)) << endl;
      cout << "same vectors: " << (a == b) << endl << endl;
    }

#if defined(__INTEL_COMPILER)
    {
    cout << "ipp radix sort: ";
    auto tic = system_clock::now();

    IppSizeL sz;
    Ipp8u *buffer;
    ippsSortRadixGetBufferSize_L(n, ipp32s, &sz);
    buffer = (Ipp8u *)malloc(sz);

    for (int j = 0; j < runs; ++j) {
      memcpy(d.data(), v.data(), sizeof(int) * n);
      ippsSortRadixAscend_32s_I_L(d.data(), n, buffer);
    }
    auto toc = system_clock::now();
      auto dur = duration<double>(toc - tic).count() - copy_time;
      cout << "\n" << dur << " sec";
      if (dur_std > 0.001) cout << "\nspeed-up: " << dur_std / double(dur);
      cout << boolalpha << "\nis_sorted: " << is_sorted(begin(d), end(d)) << endl;
      cout << "same vectors: " << (a == d) << endl << endl;
    }
#endif
  }
}