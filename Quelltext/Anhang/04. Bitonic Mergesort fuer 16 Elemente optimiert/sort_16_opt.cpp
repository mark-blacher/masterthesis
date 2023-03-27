#include <immintrin.h>

/* lade aus Array 8 int ins Vektorregister */
#define LOAD_VEC(arr) _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr))

#define STORE_VEC(arr, vec)      /* speichere Vektor ins Array */        \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr), vec)

/* Blend-Maske aus Permutations-Indices für aufsteigendes Sortieren */
#define ASC(a, b, c, d, e, f, g, h)                                      \
  (((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) |   \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0))

/* vektorisiertes compare-exchange mit Permutation */
#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h){                       \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);    \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask);   \
    __m256i min = _mm256_min_epi32(permuted, vec);                       \
    __m256i max = _mm256_max_epi32(permuted, vec);                       \
    vec = _mm256_blend_epi32(min, max, ASC(a, b, c, d, e, f, g, h));}

#define REVERSE_VEC(v) /* kehre Reihenfolge der Zahlen im Vektor um */	 \
  v = _mm256_permutevar8x32_epi32(v,                                     \
   _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));

#define MIN(x, y) ((x)+((((y)-(x))>>31)&((y)-(x)))) /* min von 2 int */
#define MAX(x, y) ((x)-((((x)-(y))>>31)&((x)-(y)))) /* max von 2 int */

#define COEX_VERTICAL(a, b){  /* berechne acht Vergleichsmodule */ 	     \
  __m256i c = a; a = _mm256_min_epi32(a, b); b = _mm256_max_epi32(c, b);}

/* shuffle 2 Vektoren, Instruktion für int fehlt, deshalb mit float */
#define SHUFFLE_2_VECS(a, b, mask)                                       \
    reinterpret_cast<__m256i>(_mm256_shuffle_ps(                         \
        reinterpret_cast<__m256>(a), reinterpret_cast<__m256>(b), mask));

/* optimiertes Sortiernetzwerk für zwei Vektoren, d. h. 16 int */
inline void sort_16(__m256i &v1, __m256i &v2) {
  COEX_VERTICAL(v1, v2);                                  /* Schritt 1 */

  v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1)); /* Schritt 2 */
  COEX_VERTICAL(v1, v2);

  auto tmp = v1;                                          /* Schritt  3 */
  v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
  COEX_VERTICAL(v1, v2);

  v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(0, 1, 2, 3)); /* Schritt  4 */
  COEX_VERTICAL(v1, v2);

  tmp = v1;                                               /* Schritt  5 */
  v1 = SHUFFLE_2_VECS(v1, v2, 0b01000100);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b11101110);
  COEX_VERTICAL(v1, v2);

  tmp = v1;                                               /* Schritt  6 */
  v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
  COEX_VERTICAL(v1, v2);

  v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(7,6,5,4,3,2,1,0));
  COEX_VERTICAL(v1, v2);                                  /* Schritt  7 */

  tmp = v1;                                               /* Schritt  8 */
  v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
  COEX_VERTICAL(v1, v2);

  tmp = v1;                                               /* Schritt  9 */
  v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
  COEX_VERTICAL(v1, v2);

  /* permutieren, damit Reihenfolge einfacher wiederherzustellen */
  v1 = _mm256_permutevar8x32_epi32(v1, _mm256_setr_epi32(0,4,1,5,6,2,7,3));
  v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(0,4,1,5,6,2,7,3));

  tmp = v1;                                              /* Schritt  10 */
  v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
  COEX_VERTICAL(v1, v2);

  /* Reihenfolge wiederherstellen */
  auto b2 = _mm256_shuffle_epi32(v2,0b10110001);
  auto b1 = _mm256_shuffle_epi32(v1,0b10110001);
  v1 = _mm256_blend_epi32(v1, b2, 0b10101010);
  v2 = _mm256_blend_epi32(b1, v2, 0b10101010);}

/* N Vektoren mergen, N ist Anzahl der Vektoren, N % 2 == 0 und N > 0
 * s = 2 bedeuted, dass jeweils zwei Vektoren bereits sortiert sind */
inline void bitonic_merge(__m256i *vecs, const int N, const int s = 2) {
  for (int t = s * 2; t < 2 * N; t *= 2) {
    for (int l = 0; l < N; l += t) {
      for (int j = MAX(l + t - N, 0); j < t/2 ; j += 2) {
        REVERSE_VEC(vecs[l + t - 1 - j]);
        REVERSE_VEC(vecs[l + t - 2 - j]);
        COEX_VERTICAL(vecs[l + j], vecs[l + t - 1 - j]);
        COEX_VERTICAL(vecs[l + j + 1], vecs[l + t - 2 - j]); }}
    for (int m = t / 2; m > 2; m /= 2) {
      for (int k = 0; k < N - m / 2; k += m) {
        const int bound = MIN((k + m / 2), N - (m / 2));
        for (int j = k; j < bound; j += 2) {
          COEX_VERTICAL(vecs[j], vecs[m / 2 + j]);
          COEX_VERTICAL(vecs[j + 1], vecs[m / 2 + j + 1]); }}}
    for (int j = 0; j < N - 1; j += 2) {
      COEX_VERTICAL(vecs[j], vecs[j + 1]); }
    for (int i = 0; i < N - 1; i += 2) {
      COEX_PERMUTE(vecs[i], 4, 5, 6, 7, 0, 1, 2, 3);
      COEX_PERMUTE(vecs[i + 1], 4, 5, 6, 7, 0, 1, 2, 3);

      auto tmp = vecs[i];
      vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
      vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
      COEX_VERTICAL(vecs[i], vecs[i + 1]);

      tmp = vecs[i];
      vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
      vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
      COEX_VERTICAL(vecs[i], vecs[i + 1]);

      tmp = vecs[i];
      vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
      vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
    }
  }
}

/* arr ist das Array, dass sortiert wird
 * n ist Anzahl der Elemente im Array
 * n  Modulo 16 muss 0 sein und n größer 0
 * buffer ist ausgerichteter Speicher mit genügend Platz für arr*/
void sort_bitonic(int *arr, __m256i* buffer, const int n) {
  for (int i = 0; i < n / 8; i += 2) {
    buffer[i] = LOAD_VEC(arr + i * 8);
    buffer[i + 1] = LOAD_VEC(arr + i * 8 + 8);
    sort_16(buffer[i], buffer[i + 1]);
  }
  bitonic_merge(buffer, n / 8, 2);
  for (int i = 0; i < n / 8; i += 2) {
    STORE_VEC(arr + i * 8, buffer[i]);
    STORE_VEC(arr + i * 8 + 8, buffer[i + 1]);
  }
}

volatile int n = 4000;
__m256i buffer[100000];

#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>
#include <algorithm>
int main(){
  const int runs = 10000; // repetitions of calculation
  std::vector<int> v(n);

  using namespace std;
  using namespace std::chrono;
  auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
  for (int i = 0; i < n; ++i) {
    v[i] = die();
  }
  auto a = v;
  auto b = v;

  double copy_time = 0;
  {
    auto tic = system_clock::now();
    for (int j = 0; j < runs; ++j) {
      memcpy(a.data(), v.data(), sizeof(int) * n);
    }
    auto toc = system_clock::now();
    copy_time = duration<double>(toc - tic).count();
  }

  {
    cout << "std::sort: ";
    auto tic = system_clock::now();
    for (int j = 0; j < runs; ++j) {
      memcpy(a.data(), v.data(), sizeof(int) * n);
      sort(begin(a), end(a));
    }
    auto toc = system_clock::now();
    auto dur = duration<double>(toc - tic).count() - copy_time;
    cout << "\n" << dur << " sec\n\n";
  }

  {
    cout << "bitonic mergesort: \n";
    auto tic = system_clock::now();

    for (int j = 0; j < runs; ++j) {
      memcpy(b.data(), v.data(), sizeof(int) * n);
      sort_bitonic(b.data(), buffer, n);
    }
    auto toc = system_clock::now();
    auto dur = duration<double>(toc - tic).count() - copy_time;
    cout << "\n" << dur << " sec";
    cout << boolalpha << "\nis_sorted: " << is_sorted(begin(b), end(b)) << endl;
    cout << "same vectors: " << (a == b) << endl << endl;
  }
}
