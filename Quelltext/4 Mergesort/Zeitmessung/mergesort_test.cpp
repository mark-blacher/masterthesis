#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "aligned_vector.h"
#include "sorting_network_int_avx2.h"

using namespace std;
using namespace std::chrono;

/*
 * Proramm ermittelt Speedup von
 * - vektorisierten nicht optimierten Mergesort
 * und
 * - vektorisierten optimierten Mergesort
 * im Vergeleich zu std::sort und IPP-Radixsort
 */

/******************************************************************************
 * vektorisierter nicht optimierter Mergesort
 */

#define ASC(a, b, c, d, e, f, g, h)                                            \
  ((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) |          \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 1)

/* vektorisiertes compare-exchange mit Permutation */
#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){                      \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);         \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask);        \
    __m256i min = _mm256_min_epi32(permuted, vec);                            \
    __m256i max = _mm256_max_epi32(permuted, vec);                            \
    vec = _mm256_blend_epi32(min, max, MASK(a, b, c, d, e, f, g, h));}

#define SORT_8(vec){                       /* sortiere aufsteigend 8 int */    \
    COEX_PERMUTE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC);         /* Schritt 1 */    \
    COEX_PERMUTE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC);         /* Schritt 2 */    \
    COEX_PERMUTE(vec, 0, 2, 1, 3, 4, 6, 5, 7, ASC);         /* Schritt 3 */    \
    COEX_PERMUTE(vec, 7, 6, 5, 4, 3, 2, 1, 0, ASC);         /* Schritt 4 */    \
    COEX_PERMUTE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC);         /* Schritt 5 */    \
    COEX_PERMUTE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC);}        /* Schritt 6 */

#define REVERSE_VEC(v){ /* kehre Reihenfolge der Zahlen im Vektor um */   \
  v = _mm256_permutevar8x32_epi32(v,                                      \
          _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));}

#define COEX_VERTICAL(a, b){ /* berechne acht Vergleichsmodule */        \
  __m256i c = a; a = _mm256_min_epi32(a, b); b = _mm256_max_epi32(c, b);}

#define LAST_3_STEPS(v){ /* letzten drei Schritte des Netzwerks */        \
  COEX_PERMUTE(v, 4, 5, 6, 7, 0, 1, 2, 3, ASC);                           \
  COEX_PERMUTE(v, 2, 3, 0, 1, 6, 7, 4, 5, ASC);                           \
  COEX_PERMUTE(v, 1, 0, 3, 2, 5, 4, 7, 6, ASC);}

#define MERGE_16(v1, v2){ /* merge zwei sortierte Vektoren */             \
  REVERSE_VEC(v2);                                                        \
  COEX_VERTICAL(v1, v2); /* Schritt 7 */                                  \
  LAST_3_STEPS(v1); LAST_3_STEPS(v2);} /* Schritte 8, 9 und 10 */

/* Arrays a und b mit a_size bzw. b_size Vektoren nach Array c mergen */
inline void merge_vectorized(__m256i *a, __m256i *b, __m256i *c,
                             const int a_size, const int b_size, const int c_size) {
  auto v_min = a[0], v_max = b[0];     /* v_min und v_max initialisieren */
  int idx_a = 1, idx_b = 1;
  for (int i = 0; i < c_size - 2; ++i) {
    MERGE_16(v_min, v_max); /* sortierte Vektoren v_min und v_max mergen */
    c[i] = v_min;                               /* abspeichern von v_min */
    if (idx_a == a_size) { v_min = b[idx_b++]; }       /* a abgearbeitet */
    else if (idx_b == b_size) { v_min = a[idx_a++]; }  /* b abgearbeitet */
    else { v_min = _mm256_extract_epi32(a[idx_a], 0) <
                   _mm256_extract_epi32(b[idx_b], 0)
                   ? a[idx_a++] :  b[idx_b++]; }}
  MERGE_16(v_min, v_max); c[c_size - 2] = v_min; c[c_size - 1] = v_max; }

/* N Vektoren sortieren, c ist zusätzlicher Speicher der Länge N */
inline void merge_sort_vectorized(__m256i *vecs, __m256i *c, const int N) {
  if (N == 1) { SORT_8(vecs[0]); return; } /* einzelnen Vektor sortieren */
  if(N > 1) {
    const int size_a = N / 2;                       /* Länge Teilfolge a */
    const int size_b = N - size_a;                  /* Länge Teilfolge b */
    merge_sort_vectorized(vecs, c, size_a);      /* Teilfolgen sortieren */
    merge_sort_vectorized(vecs + size_a, c + size_a, size_b);
    merge_vectorized(vecs, vecs + size_a, c, size_a, size_b, N); // mergen
    for (int i = 0; i < N; ++i) vecs[i] = c[i]; }} /* kopieren nach vecs */

/******************************************************************************
 * vektorisierter optimierter Mergesort
 */

#define UNPACK(vecs){           /* statt shuffle Vektoren entpacken */ \
  auto tmp = vecs[0]; auto tmp2 = vecs[2];                             \
  vecs[0] = _mm256_unpacklo_epi32(vecs[0], vecs[1]);                   \
  vecs[1] = _mm256_unpackhi_epi32(tmp, vecs[1]);                       \
  vecs[2] = _mm256_unpacklo_epi32(vecs[2], vecs[3]);                   \
  vecs[3] = _mm256_unpackhi_epi32(tmp2, vecs[3]);}

#define CV(a, b) COEX_VERTICAL(a, b) /* Aliase für weniger Code */
#define CP(vec) COEX_PERMUTE(vec, 4, 5, 6, 7, 0, 1, 2, 3, ASC);

/* 4 sortierte Vektoren in v_min mit 4 sortierten Vektoren in v_max mit
 * Bitonic Merge mergen, die vier Vektoren von v_min in c speichern */
inline void merge_2x4_vecs(__m256i *v_min, __m256i *v_max, __m256i *c) {
  REVERSE_VEC(v_max[0]); REVERSE_VEC(v_max[1]); REVERSE_VEC(v_max[2]);
  REVERSE_VEC(v_max[3]); CV(v_min[0], v_max[3]); CV(v_min[1], v_max[2]);
  CV(v_min[2], v_max[1]); CV(v_min[3], v_max[0]); CV(v_min[0], v_min[2]);
  CV(v_min[1], v_min[3]); CV(v_min[0], v_min[1]); CV(v_min[2], v_min[3]);
  CP(v_min[0]); CP(v_min[1]); CP(v_min[2]); CP(v_min[3]); UNPACK(v_min);
  CV(v_min[0], v_min[1]); CV(v_min[2], v_min[3]); UNPACK(v_min);
  CV(v_min[0], v_min[1]); CV(v_min[2], v_min[3]); UNPACK(v_min);
  c[0] = v_min[0]; c[1] = v_min[1]; c[2] = v_min[2]; c[3] = v_min[3];
  CV(v_max[0], v_max[2]); CV(v_max[1], v_max[3]); CV(v_max[0], v_max[1]);
  CV(v_max[2], v_max[3]); CP(v_max[0]); CP(v_max[1]); CP(v_max[2]);
  CP(v_max[3]); UNPACK(v_max); CV(v_max[0], v_max[1]);
  CV(v_max[2], v_max[3]); UNPACK(v_max); CV(v_max[0], v_max[1]);
  CV(v_max[2], v_max[3]); UNPACK(v_max); }

inline void merge_optimized(__m256i *a, __m256i *b, __m256i *c,
                            const int a_size, const int b_size, const int c_size) {
  __m256i *v_min = a, *v_max = b;
  int idx_a = 4, idx_b = 4;
  for (int i = 0; i < c_size - 4; i += 4) {
    merge_2x4_vecs(v_min, v_max, &c[i]);
    if (idx_a == a_size) { v_min = &b[idx_b]; b += 4; }
    else if (idx_b == b_size) { v_min = &a[idx_a]; idx_a += 4; }
    else {
      if (_mm256_extract_epi32(a[idx_a], 0) <
          _mm256_extract_epi32(b[idx_b], 0)) {
               v_min = &a[idx_a]; idx_a += 4; }
      else { v_min = &b[idx_b]; idx_b += 4; }}}
  for (int j = 0; j < 4; ++j) { c[c_size - 4 + j] = v_max[j]; }}

inline void merge_sort_optimized(__m256i *, __m256i *, int);

inline void merge_sort_helper(__m256i *vecs, __m256i *c, const int N) {
  const int size_a = N / 2 - (N / 2) % 4;
  const int size_b = N - size_a;
  merge_sort_optimized(vecs, c, size_a);
  merge_sort_optimized(vecs + size_a, c + size_a, size_b);
  merge_optimized(vecs, vecs + size_a, c, size_a, size_b, N); }

/* N % 4 == 0, sonst sortiert falsch */
inline void merge_sort_optimized(__m256i *vecs, __m256i *c, const int N) {
  if (N < 1025) { sort_int_sorting_network_aligned(vecs, N); return; }
  const int size_a = N / 2 - (N / 2) % 4;
  const int size_b = N - size_a;
  merge_sort_helper(vecs, c, size_a);
  merge_sort_helper(vecs + size_a, c + size_a, size_b);
  merge_optimized(c, c + size_a, vecs, size_a, size_b, N); }


#if defined(__INTEL_COMPILER)
#include <ipp.h>
#endif

int main() {
  bool calculate_speedups_for_diagram = false; /* auf true setzen falls Zeitmessung für Diagramm */
  if(calculate_speedups_for_diagram){
    volatile int n = 0;

    vector<int> vec_n{100000, 1000000, 10000000, 100000000, 1000000000};
    vector<int> vec_runs{1000, 200, 40, 10, 1};
    vector<double> vec_speedups(5 * 3, 0);

    int repeating = 10;
    for (int s = 0; s < repeating; ++s) {
      for (int i = 0; i < 5; ++i) {
        n = vec_n[i];
        int runs = vec_runs[i];

        vector<int, AlignmentAllocator<int, 128> > v(static_cast<unsigned long long int>(n));
        auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
        for (int r = 0; r < n; ++r) {
          v[r] = die();
        }

        vector<int, AlignmentAllocator<int, 128> > a = v;

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
          /* mergesort vectorized naive */
          auto tic = system_clock::now();
          vector<int, AlignmentAllocator<int, 128> > buffer(n);
          for (int j = 0; j < runs; ++j) {
            memcpy(a.data(), v.data(), sizeof(int) * n);
            merge_sort_vectorized(reinterpret_cast<__m256i *>(a.data()), reinterpret_cast<__m256i *>(buffer.data()), n / 8);          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          vec_speedups[i] += dur > 0.000001 ? dur_std/double(dur) : 0;
          if(!is_sorted(begin(a), end(a))){cerr << "naive failed!!!"; exit(-1);} ;
        }

        {
          /* mergesort vectorized optimized */;
          auto tic = system_clock::now();
          vector<int, AlignmentAllocator<int, 128> > buffer(n);
          for (int j = 0; j < runs; ++j) {
            memcpy(a.data(), v.data(), sizeof(int) * n);
            merge_sort_optimized(reinterpret_cast<__m256i *>(a.data()), reinterpret_cast<__m256i *>(buffer.data()), n / 8);
          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          vec_speedups[i + 5] += dur > 0.000001 ? dur_std/double(dur) : 0;
          if(!is_sorted(begin(a), end(a))){cerr << "optimized failed!!!"; exit(-1);} ;
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
          vec_speedups[i + 10] += dur > 0.00001 ? dur_std / double(dur) : 0;
          if(!is_sorted(begin(a), end(a))){cerr << "radix failed!!!"; exit(-1);} ;
#endif
        }
        cout << "n: " << n << " in repeating " << s << " finished" << endl;
      }
      cout << endl;
    }
    cout << "data_from_cpp = c(";
    for (int k = 0; k < 5 * 3 - 1; ++k) {
      cout << (vec_speedups[k] / repeating) << ", ";
    }
    cout << (vec_speedups[5 * 3 - 1] / repeating) << ")\n";
  }else {
    volatile int n = 1000000;  // size of vector
    const int runs = 100; // repetitions of calculation
    vector<int, AlignmentAllocator<int, 128> > v(static_cast<unsigned long long int>(n));

    auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
    for (int i = 0; i < n; ++i) {
      v[i] = die();
    }

    vector<int, AlignmentAllocator<int, 128> > a = v;
    vector<int, AlignmentAllocator<int, 128> > b = v;
    vector<int, AlignmentAllocator<int, 128> > c = v;
    vector<int, AlignmentAllocator<int, 128> > d = v;

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
      cout << "mergesort vectorized non optimized:";
      auto tic = system_clock::now();
      vector<int, AlignmentAllocator<int, 128> > buffer(n);
      for (int j = 0; j < runs; ++j) {
        memcpy(b.data(), v.data(), sizeof(int) * n);
        merge_sort_vectorized(reinterpret_cast<__m256i *>(b.data()), reinterpret_cast<__m256i *>(buffer.data()), n / 8);
      }
      auto toc = system_clock::now();
      auto dur = duration<double>(toc - tic).count() - copy_time;
      cout << "\n" << dur << " sec";
      if (dur_std > 0.001) cout << "\nspeed-up: " << dur_std / double(dur);
      cout << boolalpha << "\nis_sorted: " << is_sorted(begin(b), end(b)) << endl;
      cout << "same vectors: " << (a == b) << endl << endl;
    }

    {
      cout << "mergesort vectorized optimized:";
      auto tic = system_clock::now();
      vector<int, AlignmentAllocator<int, 128> > buffer(n);
      for (int j = 0; j < runs; ++j) {
        memcpy(c.data(), v.data(), sizeof(int) * n);
        merge_sort_optimized(reinterpret_cast<__m256i *>(c.data()), reinterpret_cast<__m256i *>(buffer.data()), n / 8);
      }
      auto toc = system_clock::now();
      auto dur = duration<double>(toc - tic).count() - copy_time;
      cout << "\n" << dur << " sec";
      if (dur_std > 0.001) cout << "\nspeed-up: " << dur_std / double(dur);
      cout << boolalpha << "\nis_sorted: " << is_sorted(begin(c), end(c)) << endl;
      cout << "same vectors: " << (a == c) << endl << endl;
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