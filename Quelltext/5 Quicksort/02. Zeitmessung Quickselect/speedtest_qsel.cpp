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

void test_correctness(){
  for (int j = 1; j < 4000; ++j) {
    volatile int n = j;
    vector<int> v(j);
    auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
    for (int i = 0; i < n; ++i) {
      v[i] = die();
    }
    int k = n/2;
    auto a = v;
    auto b = v;
    nth_element(begin(a), begin(a) + k, end(a));
    avx2::quickselect(b.data(), n, k);

    if(a[k] != b[k]){
      cerr << "fail " << n << endl;
      exit(-1);
    }
    sort(begin(a), begin(a) + k);
    sort(begin(a) + k + 1, end(a));
    sort(begin(b), begin(b) + k);
    sort(begin(b) + k + 1, end(b));

    if(a != b){
      cerr << "fail (same elements) " << n << endl;
      exit(-1);
    }
  }
  cout << "quickselect ist korrekt!\n" << endl;
}

int main() {
  test_correctness();
  bool calculate_speedups_for_diagram = false; /* auf true setzen falls Zeitmessung für Diagramm */
  if(calculate_speedups_for_diagram){
    volatile int n = 0;

    vector<int> vec_n{10000, 100000, 1000000, 10000000, 100000000, 1000000000};
    vector<int> vec_runs{40000, 5000, 1000, 200, 50, 5};
    vector<double> vec_speedups(6, 0);

    int repeating = 10;
    for (int s = 0; s < repeating; ++s) {
      for (int i = 0; i < 6; ++i) {
        n = vec_n[i];
        int k = n/2;
        int runs = vec_runs[i];

        vector<int> v(static_cast<unsigned long long int>(n));
        auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
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

        int tmp = 0;
        double dur_std = 0;
        {
          auto tic = system_clock::now();
          for (int j = 0; j < runs; ++j) {
            memcpy(a.data(), v.data(), sizeof(int) * n);
            nth_element(begin(a), begin(a) + k, end(a));
          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          dur_std = dur;
          tmp = a[k];
        }
        {
          /* quickselect vectorized */
          auto tic = system_clock::now();
          for (int j = 0; j < runs; ++j) {
            memcpy(a.data(), v.data(), sizeof(int) * n);
            avx2::quickselect(a.data(), n, k);
          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          vec_speedups[i] += dur > 0.000001 ? dur_std/double(dur) : 0;
          if(a[k] != tmp) cout << "fail ";
        }

        cout << "n: " << n << " in repeating " << s << " finished" << endl;
      }
      cout << endl;
    }
    cout << "data_from_cpp = c(";
    for (int l = 0; l < 6 - 1; ++l) {
      cout << (vec_speedups[l] / repeating) << ", ";
    }
    cout << (vec_speedups[6 - 1] / repeating) << ")\n";
  }else {
    volatile int n = 1000000;  // size of vector
    const int runs = 1000; // repetitions of calculation
    vector<int> v(static_cast<unsigned long long int>(n));
    int k = n/2;

    auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});

    for (int i = 0; i < n; ++i) {
      v[i] = die();
    }


    vector<int> a = v;
    vector<int> b = v;

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
      cout << "std::nth_element: ";
      auto tic = system_clock::now();
      for (int j = 0; j < runs; ++j) {
        memcpy(a.data(), v.data(), sizeof(int) * n);
        nth_element(begin(a), begin(a) + k, end(a));
      }
      auto toc = system_clock::now();
      auto dur = duration<double>(toc - tic).count() - copy_time;
      dur_std = dur;
      cout << "\n" << dur << " sec\n\n";
    }

    {
      cout << "quickselect vectorized:";
      auto tic = system_clock::now();
      for (int j = 0; j < runs; ++j) {
        memcpy(b.data(), v.data(), sizeof(int) * n);
        avx2::quickselect(b.data(), n, k);
      }
      auto toc = system_clock::now();
      auto dur = duration<double>(toc - tic).count() - copy_time;
      cout << "\n" << dur << " sec";
      if (dur_std > 0.001) cout << "\nspeed-up: " << dur_std / double(dur);
    }
  }
}

