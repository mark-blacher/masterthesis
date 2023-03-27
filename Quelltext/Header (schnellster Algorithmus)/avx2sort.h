#ifndef AVX2SORT_H
#define AVX2SORT_H

#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <vector>

/* dieser Header enthält drei vektorisierte Funktionen für den Datentyp int:
 * 1. avx2::quickselect(int *arr, int n, int k)
 * 2. avx2::quicksort(int *arr, int n)
 * 3. avx2::quicksort_omp(int *arr, int n)
 * */

namespace avx2{
namespace __internal {

#define LOAD_VEC(arr) _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr))
#define STORE_VEC(arr, vec)                                                   \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr), vec)


/* vektorisierte Sortiernetzwerke
************************************/

#define COEX_VERTICAL(a, b){                                                   \
    auto vec_tmp = a;                                                          \
    a = _mm256_min_epi32(a, b);                                                \
    b = _mm256_max_epi32(vec_tmp, b);}

/* shuffle 2 Vektoren, Instruktion für int fehlt, deshalb mit float */
#define SHUFFLE_2_VECS(a, b, mask)                                       \
    _mm256_castps_si256 (_mm256_shuffle_ps(                              \
        _mm256_castsi256_ps (a), _mm256_castsi256_ps (b), mask));

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
  v2 = _mm256_blend_epi32(b1, v2, 0b10101010);
}

#define ASC(a, b, c, d, e, f, g, h)                                    \
  (((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) | \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0))

#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){               \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);  \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask); \
    __m256i min = _mm256_min_epi32(permuted, vec);                     \
    __m256i max = _mm256_max_epi32(permuted, vec);                     \
    constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}


#define COEX_SHUFFLE(vec, a, b, c, d, e, f, g, h, MASK){               \
    constexpr int shuffle_mask = _MM_SHUFFLE(d, c, b, a);              \
    __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);        \
    __m256i min = _mm256_min_epi32(shuffled, vec);                     \
    __m256i max = _mm256_max_epi32(shuffled, vec);                     \
    constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

#define REVERSE_VEC(vec){                                              \
    vec = _mm256_permutevar8x32_epi32(                                 \
        vec, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));}

/* Sortiernetzwerk für 8 int mit compare-exchange Makros
 * (für Pivot-Berechnung in Median der Mediane) */
#define SORT_8(vec){                                                   \
  COEX_SHUFFLE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC);                           \
  COEX_SHUFFLE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC);                           \
  COEX_SHUFFLE(vec, 0, 2, 1, 3, 4, 6, 5, 7, ASC);                           \
  COEX_PERMUTE(vec, 7, 6, 5, 4, 3, 2, 1, 0, ASC);                           \
  COEX_SHUFFLE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC);                           \
  COEX_SHUFFLE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC);}

/* N Vektoren mergen, N ist Anzahl der Vektoren, N % 2 == 0 und N > 0
 * s = 2 bedeuted, dass jeweils zwei Vektoren bereits sortiert sind */
inline void bitonic_merge_16(__m256i *vecs, const int N, const int s = 2) {
  for (int t = s * 2; t < 2 * N; t *= 2) {
    for (int l = 0; l < N; l += t) {
      for (int j = std::max(l + t - N, 0); j < t/2 ; j += 2) {
        REVERSE_VEC(vecs[l + t - 1 - j]);
        REVERSE_VEC(vecs[l + t - 2 - j]);
        COEX_VERTICAL(vecs[l + j], vecs[l + t - 1 - j]);
        COEX_VERTICAL(vecs[l + j + 1], vecs[l + t - 2 - j]); }}
    for (int m = t / 2; m > 4; m /= 2) {
      for (int k = 0; k < N - m / 2; k += m) {
        const int bound = std::min((k + m / 2), N - (m / 2));
        for (int j = k; j < bound; j += 2) {
          COEX_VERTICAL(vecs[j], vecs[m / 2 + j]);
          COEX_VERTICAL(vecs[j + 1], vecs[m / 2 + j + 1]); }}}
    for (int j = 0; j < N-2; j += 4) {
      COEX_VERTICAL(vecs[j], vecs[j + 2]);
      COEX_VERTICAL(vecs[j + 1], vecs[j + 3]);
    }
    for (int j = 0; j < N; j += 2) {
      COEX_VERTICAL(vecs[j], vecs[j + 1]); }
    for (int i = 0; i < N; i += 2) {
      COEX_PERMUTE(vecs[i], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
      COEX_PERMUTE(vecs[i + 1], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
      auto tmp = vecs[i]; /* 8 Module gleichzeitig mit COEX_VERTICAL */
      vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
      vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
      COEX_VERTICAL(vecs[i], vecs[i + 1]);
      tmp = vecs[i];
      vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
      vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
      COEX_VERTICAL(vecs[i], vecs[i + 1]);
      tmp = vecs[i];
      vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
      vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]); }}}


inline void bitonic_merge_128(__m256i *vecs, const int N, const int s = 16) {
  const int remainder16 = N - N % 16;
  const int remainder8 = N - N % 8;
  for (int t = s * 2; t < 2 * N; t *= 2) {
    for (int l = 0; l < N; l += t) {
      for (int j = std::max(l + t - N, 0); j < t/2 ; j += 2) {
        REVERSE_VEC(vecs[l + t - 1 - j]);
        REVERSE_VEC(vecs[l + t - 2 - j]);
        COEX_VERTICAL(vecs[l + j], vecs[l + t - 1 - j]);
        COEX_VERTICAL(vecs[l + j + 1], vecs[l + t - 2 - j]); }}
    for (int m = t / 2; m > 16; m /= 2) {
      for (int k = 0; k < N - m / 2; k += m) {
        const int bound = std::min((k + m / 2), N - (m / 2));
        for (int j = k; j < bound; j += 2) {
          COEX_VERTICAL(vecs[j], vecs[m / 2 + j]);
          COEX_VERTICAL(vecs[j + 1], vecs[m / 2 + j + 1]); }}}
    for (int j = 0; j < remainder16; j += 16) {
      COEX_VERTICAL(vecs[j], vecs[j + 8]);
      COEX_VERTICAL(vecs[j + 1], vecs[j + 9]);
      COEX_VERTICAL(vecs[j + 2], vecs[j + 10]);
      COEX_VERTICAL(vecs[j + 3], vecs[j + 11]);
      COEX_VERTICAL(vecs[j + 4], vecs[j + 12]);
      COEX_VERTICAL(vecs[j + 5], vecs[j + 13]);
      COEX_VERTICAL(vecs[j + 6], vecs[j + 14]);
      COEX_VERTICAL(vecs[j + 7], vecs[j + 15]);
    }
    for (int j = remainder16 + 8; j < N; j += 1) {
      COEX_VERTICAL(vecs[j - 8], vecs[j]);
    }
    for (int j = 0; j < remainder8; j += 8) {
      COEX_VERTICAL(vecs[j], vecs[j + 4]);
      COEX_VERTICAL(vecs[j + 1], vecs[j + 5]);
      COEX_VERTICAL(vecs[j + 2], vecs[j + 6]);
      COEX_VERTICAL(vecs[j + 3], vecs[j + 7]);
    }
    for (int j = remainder8 + 4; j < N; j += 1) {
      COEX_VERTICAL(vecs[j - 4], vecs[j]);
    }
    for (int j = 0; j < N-2; j += 4) {
      COEX_VERTICAL(vecs[j], vecs[j + 2]);
      COEX_VERTICAL(vecs[j + 1], vecs[j + 3]);
    }
    for (int j = 0; j < N; j += 2) {
      COEX_VERTICAL(vecs[j], vecs[j + 1]); }
    for (int i = 0; i < N; i += 2) {
      COEX_PERMUTE(vecs[i], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
      COEX_PERMUTE(vecs[i + 1], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
      auto tmp = vecs[i]; /* 8 Module gleichzeitig mit COEX_VERTICAL */
      vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
      vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
      COEX_VERTICAL(vecs[i], vecs[i + 1]);
      tmp = vecs[i];
      vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
      vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
      COEX_VERTICAL(vecs[i], vecs[i + 1]);
      tmp = vecs[i];
      vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
      vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]); }}}

/* sortiere in 8 Spalten jeweils 16 int mit 60 Modulen */
#define CV(a, b) COEX_VERTICAL(a, b) /* Alias für weniger Code */
inline void sort_16_int_vertical(__m256i* vecs){
  CV(vecs[0], vecs[1]); CV(vecs[2], vecs[3]);  /* Schritt 1 */
  CV(vecs[4], vecs[5]); CV(vecs[6], vecs[7]);
  CV(vecs[8], vecs[9]); CV(vecs[10], vecs[11])
  CV(vecs[12], vecs[13]); CV(vecs[14], vecs[15])
  CV(vecs[0], vecs[2]); CV(vecs[1], vecs[3]);  /* Schritt 2 */
  CV(vecs[4], vecs[6]); CV(vecs[5], vecs[7]);
  CV(vecs[8], vecs[10]); CV(vecs[9], vecs[11]);
  CV(vecs[12], vecs[14]); CV(vecs[13], vecs[15]);
  CV(vecs[0], vecs[4]); CV(vecs[1], vecs[5]);  /* Schritt 3 */
  CV(vecs[2], vecs[6]); CV(vecs[3], vecs[7]);
  CV(vecs[8], vecs[12]); CV(vecs[9], vecs[13]);
  CV(vecs[10], vecs[14]); CV(vecs[11], vecs[15]);
  CV(vecs[0], vecs[8]); CV(vecs[1], vecs[9])   /* Schritt 4 */
  CV(vecs[2], vecs[10]); CV(vecs[3], vecs[11])
  CV(vecs[4], vecs[12]); CV(vecs[5], vecs[13])
  CV(vecs[6], vecs[14]); CV(vecs[7], vecs[15])
  CV(vecs[5], vecs[10]); CV(vecs[6], vecs[9]); /* Schritt 5 */
  CV(vecs[3], vecs[12]); CV(vecs[7], vecs[11]);
  CV(vecs[13], vecs[14]); CV(vecs[4], vecs[8]);
  CV(vecs[1], vecs[2]);
  CV(vecs[1], vecs[4]); CV(vecs[7], vecs[13]); /* Schritt 6 */
  CV(vecs[2], vecs[8]); CV(vecs[11], vecs[14]);
  CV(vecs[2], vecs[4]); CV(vecs[5], vecs[6]);  /* Schritt 7 */
  CV(vecs[9], vecs[10]); CV(vecs[11], vecs[13]);
  CV(vecs[3], vecs[8]); CV(vecs[7], vecs[12]);
  CV(vecs[3], vecs[5]); CV(vecs[6], vecs[8]);  /* Schritt 8 */
  CV(vecs[7], vecs[9]); CV(vecs[10], vecs[12]);
  CV(vecs[3], vecs[4]); CV(vecs[5], vecs[6]);  /* Schritt 9 */
  CV(vecs[7], vecs[8]); CV(vecs[9], vecs[10]);
  CV(vecs[11], vecs[12]);
  CV(vecs[6], vecs[7]); CV(vecs[8], vecs[9]); /* Schritt 10 */}

void inline merge_8_columns_with_16_elements(__m256i* vecs){
  vecs[8] = _mm256_shuffle_epi32(vecs[8], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[7], vecs[8]);
  vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[6], vecs[9]);
  vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[5], vecs[10]);
  vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[4], vecs[11]);
  vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[3], vecs[12]);
  vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[2], vecs[13]);
  vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[1], vecs[14]);
  vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[0], vecs[15]);
  vecs[4] = _mm256_shuffle_epi32(vecs[4], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[3], vecs[4]);
  vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[2], vecs[5]);
  vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[1], vecs[6]);
  vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[0], vecs[7]);
  vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[11], vecs[12]);
  vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[10], vecs[13]);
  vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[9], vecs[14]);
  vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[8], vecs[15]);
  vecs[2] = _mm256_shuffle_epi32(vecs[2], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[1], vecs[2]);
  vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[0], vecs[3]);
  vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[5], vecs[6]);
  vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[4], vecs[7]);
  vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[9], vecs[10]);
  vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[8], vecs[11]);
  vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[13], vecs[14]);
  vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[12], vecs[15]);
  vecs[1] = _mm256_shuffle_epi32(vecs[1], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[0], vecs[1]);
  vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[2], vecs[3]);
  vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[4], vecs[5]);
  vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[6], vecs[7]);
  vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[8], vecs[9]);
  vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[10], vecs[11]);
  vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[12], vecs[13]);
  vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2,3,0,1));
  CV(vecs[14], vecs[15]);
  COEX_SHUFFLE(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  vecs[8] = _mm256_shuffle_epi32(vecs[8], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[7], vecs[8]);
  vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[6], vecs[9]);
  vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[5], vecs[10]);
  vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[4], vecs[11]);
  vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[3], vecs[12]);
  vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[2], vecs[13]);
  vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[1], vecs[14]);
  vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[0], vecs[15]);
  vecs[4] = _mm256_shuffle_epi32(vecs[4], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[3], vecs[4]);
  vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[2], vecs[5]);
  vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[1], vecs[6]);
  vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[0], vecs[7]);
  vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[11], vecs[12]);
  vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[10], vecs[13]);
  vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[9], vecs[14]);
  vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[8], vecs[15]);
  vecs[2] = _mm256_shuffle_epi32(vecs[2], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[1], vecs[2]);
  vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[0], vecs[3]);
  vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[5], vecs[6]);
  vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[4], vecs[7]);
  vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[9], vecs[10]);
  vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[8], vecs[11]);
  vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[13], vecs[14]);
  vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[12], vecs[15]);
  vecs[1] = _mm256_shuffle_epi32(vecs[1], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[0], vecs[1]);
  vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[2], vecs[3]);
  vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[4], vecs[5]);
  vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[6], vecs[7]);
  vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[8], vecs[9]);
  vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[10], vecs[11]);
  vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[12], vecs[13]);
  vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0,1,2,3));
  CV(vecs[14], vecs[15]);
  COEX_SHUFFLE(vecs[0], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[1], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[2], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[3], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[4], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[5], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[6], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[7], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[8], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[9], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[10], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[11], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[12], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[13], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[14], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_SHUFFLE(vecs[15], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
  COEX_SHUFFLE(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  REVERSE_VEC(vecs[8]);
  CV(vecs[7], vecs[8]);
  REVERSE_VEC(vecs[9]);
  CV(vecs[6], vecs[9]);
  REVERSE_VEC(vecs[10]);
  CV(vecs[5], vecs[10]);
  REVERSE_VEC(vecs[11]);
  CV(vecs[4], vecs[11]);
  REVERSE_VEC(vecs[12]);
  CV(vecs[3], vecs[12]);
  REVERSE_VEC(vecs[13]);
  CV(vecs[2], vecs[13]);
  REVERSE_VEC(vecs[14]);
  CV(vecs[1], vecs[14]);
  REVERSE_VEC(vecs[15]);
  CV(vecs[0], vecs[15]);
  REVERSE_VEC(vecs[4]);
  CV(vecs[3], vecs[4]);
  REVERSE_VEC(vecs[5]);
  CV(vecs[2], vecs[5]);
  REVERSE_VEC(vecs[6]);
  CV(vecs[1], vecs[6]);
  REVERSE_VEC(vecs[7]);
  CV(vecs[0], vecs[7]);
  REVERSE_VEC(vecs[12]);
  CV(vecs[11], vecs[12]);
  REVERSE_VEC(vecs[13]);
  CV(vecs[10], vecs[13]);
  REVERSE_VEC(vecs[14]);
  CV(vecs[9], vecs[14]);
  REVERSE_VEC(vecs[15]);
  CV(vecs[8], vecs[15]);
  REVERSE_VEC(vecs[2]);
  CV(vecs[1], vecs[2]);
  REVERSE_VEC(vecs[3]);
  CV(vecs[0], vecs[3]);
  REVERSE_VEC(vecs[6]);
  CV(vecs[5], vecs[6]);
  REVERSE_VEC(vecs[7]);
  CV(vecs[4], vecs[7]);
  REVERSE_VEC(vecs[10]);
  CV(vecs[9], vecs[10]);
  REVERSE_VEC(vecs[11]);
  CV(vecs[8], vecs[11]);
  REVERSE_VEC(vecs[14]);
  CV(vecs[13], vecs[14]);
  REVERSE_VEC(vecs[15]);
  CV(vecs[12], vecs[15]);
  REVERSE_VEC(vecs[1]);
  CV(vecs[0], vecs[1]);
  REVERSE_VEC(vecs[3]);
  CV(vecs[2], vecs[3]);
  REVERSE_VEC(vecs[5]);
  CV(vecs[4], vecs[5]);
  REVERSE_VEC(vecs[7]);
  CV(vecs[6], vecs[7]);
  REVERSE_VEC(vecs[9]);
  CV(vecs[8], vecs[9]);
  REVERSE_VEC(vecs[11]);
  CV(vecs[10], vecs[11]);
  REVERSE_VEC(vecs[13]);
  CV(vecs[12], vecs[13]);
  REVERSE_VEC(vecs[15]);
  CV(vecs[14], vecs[15]);
  COEX_PERMUTE(vecs[0], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[0], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[1], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[1], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[2], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[2], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[3], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[3], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[4], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[4], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[5], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[5], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[6], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[6], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[7], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[7], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[8], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[8], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[9], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[9], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[10], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[10], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[11], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[11], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[12], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[12], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[13], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[13], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[14], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[14], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
  COEX_PERMUTE(vecs[15], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
  COEX_SHUFFLE(vecs[15], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
  COEX_SHUFFLE(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
}

inline void sort_int_sorting_network(int *arr, int *buff, int n) {
  if(n < 2) return;
  __m256i *buffer = reinterpret_cast<__m256i *>(buff);

  const auto remainder = int(n % 8 ? n % 8 : 8);
  const int idx_max_pad = n - remainder;
  const auto mask = _mm256_add_epi32(_mm256_set1_epi32(-remainder), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
  auto max_pad_vec = _mm256_blendv_epi8(_mm256_set1_epi32(INT32_MAX),_mm256_maskload_epi32(arr + idx_max_pad, mask), mask);

  for (int i = 0; i < idx_max_pad/8; ++i) {
    buffer[i] = LOAD_VEC(arr + i * 8);
  }
  buffer[idx_max_pad/8] = max_pad_vec;
  buffer[idx_max_pad/8 + 1] = _mm256_set1_epi32(INT32_MAX);

  const int N = ((idx_max_pad % 16 == 0) * 8 + idx_max_pad + 8 )/8;

  for (int j = 0; j < N - N % 16; j+=16) {
    sort_16_int_vertical(buffer + j);
    merge_8_columns_with_16_elements(buffer + j);
  }
  for (int i = N - N % 16; i < N; i += 2) {
    sort_16(buffer[i], buffer[i + 1]);
  }
  bitonic_merge_16(buffer + N - N % 16, N % 16, 2);
  bitonic_merge_128(buffer, N, 16);
  for (int i = 0; i < idx_max_pad/8; i += 1) {
    STORE_VEC(arr + i * 8, buffer[i]);
  }
  _mm256_maskstore_epi32(arr + idx_max_pad, mask, buffer[idx_max_pad/8]);
}
/* Ende Sortiernetzwerke
*********************************************/

/*** vektorisierter Quicksort
**************************************/

/* Permutationsmasken für Quicksort */
const __m256i permutation_masks[256] = {_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0),
                                        _mm256_setr_epi32(0, 2, 3, 4, 5, 6, 7, 1),
                                        _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1),
                                        _mm256_setr_epi32(0, 1, 3, 4, 5, 6, 7, 2),
                                        _mm256_setr_epi32(1, 3, 4, 5, 6, 7, 0, 2),
                                        _mm256_setr_epi32(0, 3, 4, 5, 6, 7, 1, 2),
                                        _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2),
                                        _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 7, 3),
                                        _mm256_setr_epi32(1, 2, 4, 5, 6, 7, 0, 3),
                                        _mm256_setr_epi32(0, 2, 4, 5, 6, 7, 1, 3),
                                        _mm256_setr_epi32(2, 4, 5, 6, 7, 0, 1, 3),
                                        _mm256_setr_epi32(0, 1, 4, 5, 6, 7, 2, 3),
                                        _mm256_setr_epi32(1, 4, 5, 6, 7, 0, 2, 3),
                                        _mm256_setr_epi32(0, 4, 5, 6, 7, 1, 2, 3),
                                        _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3),
                                        _mm256_setr_epi32(0, 1, 2, 3, 5, 6, 7, 4),
                                        _mm256_setr_epi32(1, 2, 3, 5, 6, 7, 0, 4),
                                        _mm256_setr_epi32(0, 2, 3, 5, 6, 7, 1, 4),
                                        _mm256_setr_epi32(2, 3, 5, 6, 7, 0, 1, 4),
                                        _mm256_setr_epi32(0, 1, 3, 5, 6, 7, 2, 4),
                                        _mm256_setr_epi32(1, 3, 5, 6, 7, 0, 2, 4),
                                        _mm256_setr_epi32(0, 3, 5, 6, 7, 1, 2, 4),
                                        _mm256_setr_epi32(3, 5, 6, 7, 0, 1, 2, 4),
                                        _mm256_setr_epi32(0, 1, 2, 5, 6, 7, 3, 4),
                                        _mm256_setr_epi32(1, 2, 5, 6, 7, 0, 3, 4),
                                        _mm256_setr_epi32(0, 2, 5, 6, 7, 1, 3, 4),
                                        _mm256_setr_epi32(2, 5, 6, 7, 0, 1, 3, 4),
                                        _mm256_setr_epi32(0, 1, 5, 6, 7, 2, 3, 4),
                                        _mm256_setr_epi32(1, 5, 6, 7, 0, 2, 3, 4),
                                        _mm256_setr_epi32(0, 5, 6, 7, 1, 2, 3, 4),
                                        _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 6, 7, 5),
                                        _mm256_setr_epi32(1, 2, 3, 4, 6, 7, 0, 5),
                                        _mm256_setr_epi32(0, 2, 3, 4, 6, 7, 1, 5),
                                        _mm256_setr_epi32(2, 3, 4, 6, 7, 0, 1, 5),
                                        _mm256_setr_epi32(0, 1, 3, 4, 6, 7, 2, 5),
                                        _mm256_setr_epi32(1, 3, 4, 6, 7, 0, 2, 5),
                                        _mm256_setr_epi32(0, 3, 4, 6, 7, 1, 2, 5),
                                        _mm256_setr_epi32(3, 4, 6, 7, 0, 1, 2, 5),
                                        _mm256_setr_epi32(0, 1, 2, 4, 6, 7, 3, 5),
                                        _mm256_setr_epi32(1, 2, 4, 6, 7, 0, 3, 5),
                                        _mm256_setr_epi32(0, 2, 4, 6, 7, 1, 3, 5),
                                        _mm256_setr_epi32(2, 4, 6, 7, 0, 1, 3, 5),
                                        _mm256_setr_epi32(0, 1, 4, 6, 7, 2, 3, 5),
                                        _mm256_setr_epi32(1, 4, 6, 7, 0, 2, 3, 5),
                                        _mm256_setr_epi32(0, 4, 6, 7, 1, 2, 3, 5),
                                        _mm256_setr_epi32(4, 6, 7, 0, 1, 2, 3, 5),
                                        _mm256_setr_epi32(0, 1, 2, 3, 6, 7, 4, 5),
                                        _mm256_setr_epi32(1, 2, 3, 6, 7, 0, 4, 5),
                                        _mm256_setr_epi32(0, 2, 3, 6, 7, 1, 4, 5),
                                        _mm256_setr_epi32(2, 3, 6, 7, 0, 1, 4, 5),
                                        _mm256_setr_epi32(0, 1, 3, 6, 7, 2, 4, 5),
                                        _mm256_setr_epi32(1, 3, 6, 7, 0, 2, 4, 5),
                                        _mm256_setr_epi32(0, 3, 6, 7, 1, 2, 4, 5),
                                        _mm256_setr_epi32(3, 6, 7, 0, 1, 2, 4, 5),
                                        _mm256_setr_epi32(0, 1, 2, 6, 7, 3, 4, 5),
                                        _mm256_setr_epi32(1, 2, 6, 7, 0, 3, 4, 5),
                                        _mm256_setr_epi32(0, 2, 6, 7, 1, 3, 4, 5),
                                        _mm256_setr_epi32(2, 6, 7, 0, 1, 3, 4, 5),
                                        _mm256_setr_epi32(0, 1, 6, 7, 2, 3, 4, 5),
                                        _mm256_setr_epi32(1, 6, 7, 0, 2, 3, 4, 5),
                                        _mm256_setr_epi32(0, 6, 7, 1, 2, 3, 4, 5),
                                        _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 7, 6),
                                        _mm256_setr_epi32(1, 2, 3, 4, 5, 7, 0, 6),
                                        _mm256_setr_epi32(0, 2, 3, 4, 5, 7, 1, 6),
                                        _mm256_setr_epi32(2, 3, 4, 5, 7, 0, 1, 6),
                                        _mm256_setr_epi32(0, 1, 3, 4, 5, 7, 2, 6),
                                        _mm256_setr_epi32(1, 3, 4, 5, 7, 0, 2, 6),
                                        _mm256_setr_epi32(0, 3, 4, 5, 7, 1, 2, 6),
                                        _mm256_setr_epi32(3, 4, 5, 7, 0, 1, 2, 6),
                                        _mm256_setr_epi32(0, 1, 2, 4, 5, 7, 3, 6),
                                        _mm256_setr_epi32(1, 2, 4, 5, 7, 0, 3, 6),
                                        _mm256_setr_epi32(0, 2, 4, 5, 7, 1, 3, 6),
                                        _mm256_setr_epi32(2, 4, 5, 7, 0, 1, 3, 6),
                                        _mm256_setr_epi32(0, 1, 4, 5, 7, 2, 3, 6),
                                        _mm256_setr_epi32(1, 4, 5, 7, 0, 2, 3, 6),
                                        _mm256_setr_epi32(0, 4, 5, 7, 1, 2, 3, 6),
                                        _mm256_setr_epi32(4, 5, 7, 0, 1, 2, 3, 6),
                                        _mm256_setr_epi32(0, 1, 2, 3, 5, 7, 4, 6),
                                        _mm256_setr_epi32(1, 2, 3, 5, 7, 0, 4, 6),
                                        _mm256_setr_epi32(0, 2, 3, 5, 7, 1, 4, 6),
                                        _mm256_setr_epi32(2, 3, 5, 7, 0, 1, 4, 6),
                                        _mm256_setr_epi32(0, 1, 3, 5, 7, 2, 4, 6),
                                        _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6),
                                        _mm256_setr_epi32(0, 3, 5, 7, 1, 2, 4, 6),
                                        _mm256_setr_epi32(3, 5, 7, 0, 1, 2, 4, 6),
                                        _mm256_setr_epi32(0, 1, 2, 5, 7, 3, 4, 6),
                                        _mm256_setr_epi32(1, 2, 5, 7, 0, 3, 4, 6),
                                        _mm256_setr_epi32(0, 2, 5, 7, 1, 3, 4, 6),
                                        _mm256_setr_epi32(2, 5, 7, 0, 1, 3, 4, 6),
                                        _mm256_setr_epi32(0, 1, 5, 7, 2, 3, 4, 6),
                                        _mm256_setr_epi32(1, 5, 7, 0, 2, 3, 4, 6),
                                        _mm256_setr_epi32(0, 5, 7, 1, 2, 3, 4, 6),
                                        _mm256_setr_epi32(5, 7, 0, 1, 2, 3, 4, 6),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 7, 5, 6),
                                        _mm256_setr_epi32(1, 2, 3, 4, 7, 0, 5, 6),
                                        _mm256_setr_epi32(0, 2, 3, 4, 7, 1, 5, 6),
                                        _mm256_setr_epi32(2, 3, 4, 7, 0, 1, 5, 6),
                                        _mm256_setr_epi32(0, 1, 3, 4, 7, 2, 5, 6),
                                        _mm256_setr_epi32(1, 3, 4, 7, 0, 2, 5, 6),
                                        _mm256_setr_epi32(0, 3, 4, 7, 1, 2, 5, 6),
                                        _mm256_setr_epi32(3, 4, 7, 0, 1, 2, 5, 6),
                                        _mm256_setr_epi32(0, 1, 2, 4, 7, 3, 5, 6),
                                        _mm256_setr_epi32(1, 2, 4, 7, 0, 3, 5, 6),
                                        _mm256_setr_epi32(0, 2, 4, 7, 1, 3, 5, 6),
                                        _mm256_setr_epi32(2, 4, 7, 0, 1, 3, 5, 6),
                                        _mm256_setr_epi32(0, 1, 4, 7, 2, 3, 5, 6),
                                        _mm256_setr_epi32(1, 4, 7, 0, 2, 3, 5, 6),
                                        _mm256_setr_epi32(0, 4, 7, 1, 2, 3, 5, 6),
                                        _mm256_setr_epi32(4, 7, 0, 1, 2, 3, 5, 6),
                                        _mm256_setr_epi32(0, 1, 2, 3, 7, 4, 5, 6),
                                        _mm256_setr_epi32(1, 2, 3, 7, 0, 4, 5, 6),
                                        _mm256_setr_epi32(0, 2, 3, 7, 1, 4, 5, 6),
                                        _mm256_setr_epi32(2, 3, 7, 0, 1, 4, 5, 6),
                                        _mm256_setr_epi32(0, 1, 3, 7, 2, 4, 5, 6),
                                        _mm256_setr_epi32(1, 3, 7, 0, 2, 4, 5, 6),
                                        _mm256_setr_epi32(0, 3, 7, 1, 2, 4, 5, 6),
                                        _mm256_setr_epi32(3, 7, 0, 1, 2, 4, 5, 6),
                                        _mm256_setr_epi32(0, 1, 2, 7, 3, 4, 5, 6),
                                        _mm256_setr_epi32(1, 2, 7, 0, 3, 4, 5, 6),
                                        _mm256_setr_epi32(0, 2, 7, 1, 3, 4, 5, 6),
                                        _mm256_setr_epi32(2, 7, 0, 1, 3, 4, 5, 6),
                                        _mm256_setr_epi32(0, 1, 7, 2, 3, 4, 5, 6),
                                        _mm256_setr_epi32(1, 7, 0, 2, 3, 4, 5, 6),
                                        _mm256_setr_epi32(0, 7, 1, 2, 3, 4, 5, 6),
                                        _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 0, 7),
                                        _mm256_setr_epi32(0, 2, 3, 4, 5, 6, 1, 7),
                                        _mm256_setr_epi32(2, 3, 4, 5, 6, 0, 1, 7),
                                        _mm256_setr_epi32(0, 1, 3, 4, 5, 6, 2, 7),
                                        _mm256_setr_epi32(1, 3, 4, 5, 6, 0, 2, 7),
                                        _mm256_setr_epi32(0, 3, 4, 5, 6, 1, 2, 7),
                                        _mm256_setr_epi32(3, 4, 5, 6, 0, 1, 2, 7),
                                        _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7),
                                        _mm256_setr_epi32(1, 2, 4, 5, 6, 0, 3, 7),
                                        _mm256_setr_epi32(0, 2, 4, 5, 6, 1, 3, 7),
                                        _mm256_setr_epi32(2, 4, 5, 6, 0, 1, 3, 7),
                                        _mm256_setr_epi32(0, 1, 4, 5, 6, 2, 3, 7),
                                        _mm256_setr_epi32(1, 4, 5, 6, 0, 2, 3, 7),
                                        _mm256_setr_epi32(0, 4, 5, 6, 1, 2, 3, 7),
                                        _mm256_setr_epi32(4, 5, 6, 0, 1, 2, 3, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 5, 6, 4, 7),
                                        _mm256_setr_epi32(1, 2, 3, 5, 6, 0, 4, 7),
                                        _mm256_setr_epi32(0, 2, 3, 5, 6, 1, 4, 7),
                                        _mm256_setr_epi32(2, 3, 5, 6, 0, 1, 4, 7),
                                        _mm256_setr_epi32(0, 1, 3, 5, 6, 2, 4, 7),
                                        _mm256_setr_epi32(1, 3, 5, 6, 0, 2, 4, 7),
                                        _mm256_setr_epi32(0, 3, 5, 6, 1, 2, 4, 7),
                                        _mm256_setr_epi32(3, 5, 6, 0, 1, 2, 4, 7),
                                        _mm256_setr_epi32(0, 1, 2, 5, 6, 3, 4, 7),
                                        _mm256_setr_epi32(1, 2, 5, 6, 0, 3, 4, 7),
                                        _mm256_setr_epi32(0, 2, 5, 6, 1, 3, 4, 7),
                                        _mm256_setr_epi32(2, 5, 6, 0, 1, 3, 4, 7),
                                        _mm256_setr_epi32(0, 1, 5, 6, 2, 3, 4, 7),
                                        _mm256_setr_epi32(1, 5, 6, 0, 2, 3, 4, 7),
                                        _mm256_setr_epi32(0, 5, 6, 1, 2, 3, 4, 7),
                                        _mm256_setr_epi32(5, 6, 0, 1, 2, 3, 4, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 6, 5, 7),
                                        _mm256_setr_epi32(1, 2, 3, 4, 6, 0, 5, 7),
                                        _mm256_setr_epi32(0, 2, 3, 4, 6, 1, 5, 7),
                                        _mm256_setr_epi32(2, 3, 4, 6, 0, 1, 5, 7),
                                        _mm256_setr_epi32(0, 1, 3, 4, 6, 2, 5, 7),
                                        _mm256_setr_epi32(1, 3, 4, 6, 0, 2, 5, 7),
                                        _mm256_setr_epi32(0, 3, 4, 6, 1, 2, 5, 7),
                                        _mm256_setr_epi32(3, 4, 6, 0, 1, 2, 5, 7),
                                        _mm256_setr_epi32(0, 1, 2, 4, 6, 3, 5, 7),
                                        _mm256_setr_epi32(1, 2, 4, 6, 0, 3, 5, 7),
                                        _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7),
                                        _mm256_setr_epi32(2, 4, 6, 0, 1, 3, 5, 7),
                                        _mm256_setr_epi32(0, 1, 4, 6, 2, 3, 5, 7),
                                        _mm256_setr_epi32(1, 4, 6, 0, 2, 3, 5, 7),
                                        _mm256_setr_epi32(0, 4, 6, 1, 2, 3, 5, 7),
                                        _mm256_setr_epi32(4, 6, 0, 1, 2, 3, 5, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 6, 4, 5, 7),
                                        _mm256_setr_epi32(1, 2, 3, 6, 0, 4, 5, 7),
                                        _mm256_setr_epi32(0, 2, 3, 6, 1, 4, 5, 7),
                                        _mm256_setr_epi32(2, 3, 6, 0, 1, 4, 5, 7),
                                        _mm256_setr_epi32(0, 1, 3, 6, 2, 4, 5, 7),
                                        _mm256_setr_epi32(1, 3, 6, 0, 2, 4, 5, 7),
                                        _mm256_setr_epi32(0, 3, 6, 1, 2, 4, 5, 7),
                                        _mm256_setr_epi32(3, 6, 0, 1, 2, 4, 5, 7),
                                        _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7),
                                        _mm256_setr_epi32(1, 2, 6, 0, 3, 4, 5, 7),
                                        _mm256_setr_epi32(0, 2, 6, 1, 3, 4, 5, 7),
                                        _mm256_setr_epi32(2, 6, 0, 1, 3, 4, 5, 7),
                                        _mm256_setr_epi32(0, 1, 6, 2, 3, 4, 5, 7),
                                        _mm256_setr_epi32(1, 6, 0, 2, 3, 4, 5, 7),
                                        _mm256_setr_epi32(0, 6, 1, 2, 3, 4, 5, 7),
                                        _mm256_setr_epi32(6, 0, 1, 2, 3, 4, 5, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 4, 5, 0, 6, 7),
                                        _mm256_setr_epi32(0, 2, 3, 4, 5, 1, 6, 7),
                                        _mm256_setr_epi32(2, 3, 4, 5, 0, 1, 6, 7),
                                        _mm256_setr_epi32(0, 1, 3, 4, 5, 2, 6, 7),
                                        _mm256_setr_epi32(1, 3, 4, 5, 0, 2, 6, 7),
                                        _mm256_setr_epi32(0, 3, 4, 5, 1, 2, 6, 7),
                                        _mm256_setr_epi32(3, 4, 5, 0, 1, 2, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 4, 5, 3, 6, 7),
                                        _mm256_setr_epi32(1, 2, 4, 5, 0, 3, 6, 7),
                                        _mm256_setr_epi32(0, 2, 4, 5, 1, 3, 6, 7),
                                        _mm256_setr_epi32(2, 4, 5, 0, 1, 3, 6, 7),
                                        _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7),
                                        _mm256_setr_epi32(1, 4, 5, 0, 2, 3, 6, 7),
                                        _mm256_setr_epi32(0, 4, 5, 1, 2, 3, 6, 7),
                                        _mm256_setr_epi32(4, 5, 0, 1, 2, 3, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 5, 4, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 5, 0, 4, 6, 7),
                                        _mm256_setr_epi32(0, 2, 3, 5, 1, 4, 6, 7),
                                        _mm256_setr_epi32(2, 3, 5, 0, 1, 4, 6, 7),
                                        _mm256_setr_epi32(0, 1, 3, 5, 2, 4, 6, 7),
                                        _mm256_setr_epi32(1, 3, 5, 0, 2, 4, 6, 7),
                                        _mm256_setr_epi32(0, 3, 5, 1, 2, 4, 6, 7),
                                        _mm256_setr_epi32(3, 5, 0, 1, 2, 4, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 5, 3, 4, 6, 7),
                                        _mm256_setr_epi32(1, 2, 5, 0, 3, 4, 6, 7),
                                        _mm256_setr_epi32(0, 2, 5, 1, 3, 4, 6, 7),
                                        _mm256_setr_epi32(2, 5, 0, 1, 3, 4, 6, 7),
                                        _mm256_setr_epi32(0, 1, 5, 2, 3, 4, 6, 7),
                                        _mm256_setr_epi32(1, 5, 0, 2, 3, 4, 6, 7),
                                        _mm256_setr_epi32(0, 5, 1, 2, 3, 4, 6, 7),
                                        _mm256_setr_epi32(5, 0, 1, 2, 3, 4, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 4, 0, 5, 6, 7),
                                        _mm256_setr_epi32(0, 2, 3, 4, 1, 5, 6, 7),
                                        _mm256_setr_epi32(2, 3, 4, 0, 1, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 3, 4, 2, 5, 6, 7),
                                        _mm256_setr_epi32(1, 3, 4, 0, 2, 5, 6, 7),
                                        _mm256_setr_epi32(0, 3, 4, 1, 2, 5, 6, 7),
                                        _mm256_setr_epi32(3, 4, 0, 1, 2, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 4, 3, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 4, 0, 3, 5, 6, 7),
                                        _mm256_setr_epi32(0, 2, 4, 1, 3, 5, 6, 7),
                                        _mm256_setr_epi32(2, 4, 0, 1, 3, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 4, 2, 3, 5, 6, 7),
                                        _mm256_setr_epi32(1, 4, 0, 2, 3, 5, 6, 7),
                                        _mm256_setr_epi32(0, 4, 1, 2, 3, 5, 6, 7),
                                        _mm256_setr_epi32(4, 0, 1, 2, 3, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 3, 0, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 2, 3, 1, 4, 5, 6, 7),
                                        _mm256_setr_epi32(2, 3, 0, 1, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 3, 2, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 3, 0, 2, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 3, 1, 2, 4, 5, 6, 7),
                                        _mm256_setr_epi32(3, 0, 1, 2, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 2, 0, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 2, 1, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(2, 0, 1, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(1, 0, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)};

/* einzelnen Vektor partitionieren, zurückgeben wie viele Werte größer als Pivot,
 * kleinsten und größten Werte in smallest_vec bzw. biggest_vec updaten*/
inline int partition_vec(__m256i &curr_vec, const __m256i &pivot_vec,
                         __m256i &smallest_vec, __m256i &biggest_vec){
  /* welche Elemente sind größer als Pivot */
  __m256i compared = _mm256_cmpgt_epi32(curr_vec, pivot_vec);
  /* kleinsten und größten Werte des Arrays updaten */
  smallest_vec = _mm256_min_epi32(curr_vec, smallest_vec);
  biggest_vec = _mm256_max_epi32(curr_vec, biggest_vec);
  /* höchstes Bit aus jeden Integer des Vektors extrahieren */
  int mm = _mm256_movemask_ps(reinterpret_cast<__m256>(compared));
  /* wie viele einsen, jede 1 steht für Element größer als Pivot */
  int amount_gt_pivot = _mm_popcnt_u32((mm));
  /* Elemente größer als Pivot nach rechts, kleiner gleich nach links */
  curr_vec = _mm256_permutevar8x32_epi32(curr_vec, permutation_masks[mm]);
  /* zurückgeben wie viele Elemente größer als Pivot */
  return amount_gt_pivot; }

inline int calc_min(__m256i vec){ /* Minimum aus 8 int */
  auto perm_mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  vec = _mm256_min_epi32(vec, _mm256_permutevar8x32_epi32(vec, perm_mask));
  vec = _mm256_min_epi32(vec, _mm256_shuffle_epi32(vec, 0b10110001));
  vec = _mm256_min_epi32(vec, _mm256_shuffle_epi32(vec, 0b01001110));
  return _mm256_extract_epi32(vec, 0); }

inline int calc_max(__m256i vec){ /* Maximum aus 8 int */
  auto perm_mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  vec = _mm256_max_epi32(vec, _mm256_permutevar8x32_epi32(vec, perm_mask));
  vec = _mm256_max_epi32(vec, _mm256_shuffle_epi32(vec, 0b10110001));
  vec = _mm256_max_epi32(vec, _mm256_shuffle_epi32(vec, 0b01001110));
  return _mm256_extract_epi32(vec, 0); }

inline int partition_vectorized_8(int *arr, int left, int right,
                                  int pivot, int &smallest, int &biggest) {
  for (int i = (right - left) % 8; i > 0; --i) { /* Array kürzen */
    smallest = std::min(smallest, arr[left]); biggest = std::max(biggest, arr[left]);
    if (arr[left] > pivot) { std::swap(arr[left], arr[--right]); }
    else { ++left; }}

  if(left == right) return left; /* weniger wie 8 Elemente im Array */

  auto pivot_vec = _mm256_set1_epi32(pivot); /* Vektor befüllt mit Pivot */
  auto sv = _mm256_set1_epi32(smallest); /* Vektor für smallest */
  auto bv = _mm256_set1_epi32(biggest); /* Vektor für biggest */

  if(right - left == 8){ /* falls 8 Elemente übrig nach Kürzung */
    auto v = LOAD_VEC(arr + left);
    int amount_gt_pivot = partition_vec(v, pivot_vec, sv, bv);
    STORE_VEC(arr + left, v);
    smallest = calc_min(sv); biggest = calc_max(bv);
    return left + (8 - amount_gt_pivot); }

  /* ersten und letzten 8 Werte werden zum Schluss partitioniert */
  auto vec_left = LOAD_VEC(arr + left); /* ersten 8 Werte */
  auto vec_right = LOAD_VEC(arr + (right - 8)); /* letzten 8 Werte  */
  /* Positionen an denen die Vektoren gespeichert werden */
  int r_store = right - 8; /* rechte Position zum Speichern */
  int l_store = left; /* linke Position zum Speichern */
  /* Positionen an denen Vektoren geladen werden */
  left += 8; /* erhöhen, da ersten 8 Elemente zwischengespeichert */
  right -= 8; /* verringern, da letzten 8 Elemente zwischengespeichert */

  while(right - left != 0) { /* 8 Elemente pro Iteration partitionieren */
    __m256i curr_vec; /* Vektor der partitioniert wird */
    /* wenn weniger Elemente auf rechter Seite des Arrays gespeichert, dann
     * nächster Vektor von rechter Seite, sonst von linker Seite */
    if((r_store + 8) - right < left - l_store){
      right -= 8; curr_vec = LOAD_VEC(arr + right); }
    else { curr_vec = LOAD_VEC(arr + left); left += 8; }
    /* aktuellen Vektor partitionieren und auf beiden Seiten speichern */
    int amount_gt_pivot = partition_vec(curr_vec, pivot_vec, sv, bv);;
    STORE_VEC(arr + l_store, curr_vec); STORE_VEC(arr + r_store, curr_vec);
    /* Positionen an denen die Vektoren gespeichert werden updaten */
    r_store -= amount_gt_pivot; l_store += (8 - amount_gt_pivot); }

  /* vec_left partitionieren und speichern */
  int amount_gt_pivot = partition_vec(vec_left, pivot_vec, sv, bv);
  STORE_VEC(arr + l_store, vec_left); STORE_VEC(arr + r_store, vec_left);
  l_store += (8 - amount_gt_pivot);
  /* vec_right partitionieren und speichern */
  amount_gt_pivot = partition_vec(vec_right, pivot_vec, sv, bv);
  STORE_VEC(arr + l_store, vec_right);
  l_store += (8 - amount_gt_pivot);

  smallest = calc_min(sv); /* kleinsten Wert im Vektor */
  biggest = calc_max(bv); /* größter Wert im Vektor */
  return l_store; }

inline int partition_vectorized_64(int *arr, int left, int right,
                                   int pivot, int &smallest, int &biggest) {
  if (right - left < 129) { /* nicht optimieren wenn weniger als 129 Elemente */
    return partition_vectorized_8(arr, left, right, pivot, smallest, biggest); }

  for (int i = (right - left) % 8; i > 0; --i) { /* Array kürzen */
    smallest = std::min(smallest, arr[left]); biggest = std::max(biggest, arr[left]);
    if (arr[left] > pivot) { std::swap(arr[left], arr[--right]); }
    else { ++left; }}

  auto pivot_vec = _mm256_set1_epi32(pivot); /* Vektor befüllt mit Pivot */
  auto sv = _mm256_set1_epi32(smallest); /* Vektor für smallest */
  auto bv = _mm256_set1_epi32(biggest); /* Vektor für biggest */

  /* vektorweise Array kürzen bis Anzahl der Elemente ohne Rest durch 64 teilbar */
  for (int i = ((right - left) % 64) / 8; i > 0; --i) {
    __m256i vec_L = LOAD_VEC(arr + left);
    /* wie in Funktion partition_vec */
    __m256i compared = _mm256_cmpgt_epi32(vec_L, pivot_vec);
    sv = _mm256_min_epi32(vec_L, sv); bv = _mm256_max_epi32(vec_L, bv);
    int mm = _mm256_movemask_ps(reinterpret_cast<__m256>(compared));
    int amount_gt_pivot = _mm_popcnt_u32((mm));
    __m256i permuted = _mm256_permutevar8x32_epi32(vec_L, permutation_masks[mm]);

    /* dies ist eine langsamere Möglichkeit vektorisiert ein Array zu partitionieren */
    __m256i blend_mask = _mm256_cmpgt_epi32(permuted, pivot_vec);
    __m256i vec_R = LOAD_VEC(arr + right - 8);
    __m256i vec_L_new = _mm256_blendv_epi8(permuted, vec_R, blend_mask);
    __m256i vec_R_new = _mm256_blendv_epi8(vec_R, permuted, blend_mask);
    STORE_VEC(arr + left, vec_L_new); STORE_VEC(arr + right - 8, vec_R_new);
    left += (8 - amount_gt_pivot); right -= amount_gt_pivot; }

  /* jeweils 8 Vektoren von beiden Seiten des Arrays zwischengespeichern*/
  auto vec_left = LOAD_VEC(arr + left), vec_left2 = LOAD_VEC(arr + left + 8);
  auto vec_left3 = LOAD_VEC(arr + left + 16), vec_left4 = LOAD_VEC(arr + left + 24);
  auto vec_left5 = LOAD_VEC(arr + left + 32), vec_left6 = LOAD_VEC(arr + left + 40);
  auto vec_left7 = LOAD_VEC(arr + left + 48), vec_left8 = LOAD_VEC(arr + left + 56);
  auto vec_right = LOAD_VEC(arr + (right - 64)), vec_right2 = LOAD_VEC(arr + (right - 56));
  auto vec_right3 = LOAD_VEC(arr + (right - 48)), vec_right4 = LOAD_VEC(arr + (right - 40));
  auto vec_right5 = LOAD_VEC(arr + (right - 32)), vec_right6 = LOAD_VEC(arr + (right - 24));
  auto vec_right7 = LOAD_VEC(arr + (right - 16)), vec_right8 = LOAD_VEC(arr + (right - 8));

  /* Positionen an denen die Vektoren gespeichert werden */
  int r_store = right - 64; /* rechte Position zum Speichern */
  int l_store = left; /* linke Position zum Speichern */
  /* Positionen an denen Vektoren geladen werden */
  left += 64; /* erhöhen, da ersten 64 Elemente zwischengespeichert */
  right -= 64; /* verringern, da letzten 64 Elemente zwischengespeichert */

  while (right - left != 0) { /* 64 Elemente pro Iteration partitionieren */
    __m256i curr_vec, curr_vec2, curr_vec3, curr_vec4, curr_vec5, curr_vec6, curr_vec7, curr_vec8;

    /* wenn weniger Elemente auf rechter Seite des Arrays gespeichert, dann
    * nächsten 8 Vektoren von rechter Seite, sonst von linker Seite laden */
    if ((r_store + 64) - right < left - l_store) {
      right -= 64;
      curr_vec = LOAD_VEC(arr + right); curr_vec2 = LOAD_VEC(arr + right + 8);
      curr_vec3 = LOAD_VEC(arr + right + 16); curr_vec4 = LOAD_VEC(arr + right + 24);
      curr_vec5 = LOAD_VEC(arr + right + 32); curr_vec6 = LOAD_VEC(arr + right + 40);
      curr_vec7 = LOAD_VEC(arr + right + 48); curr_vec8 = LOAD_VEC(arr + right + 56); }
    else {
      curr_vec = LOAD_VEC(arr + left); curr_vec2 = LOAD_VEC(arr + left + 8);
      curr_vec3 = LOAD_VEC(arr + left + 16); curr_vec4 = LOAD_VEC(arr + left + 24);
      curr_vec5 = LOAD_VEC(arr + left + 32); curr_vec6 = LOAD_VEC(arr + left + 40);
      curr_vec7 = LOAD_VEC(arr + left + 48); curr_vec8 = LOAD_VEC(arr + left + 56);
      left += 64; }

    /* 8 Vektoren partitionieren und auf beiden Seiten des Arrays speichern */
    int amount_gt_pivot = partition_vec(curr_vec, pivot_vec, sv, bv);
    int amount_gt_pivot2 = partition_vec(curr_vec2, pivot_vec, sv, bv);
    int amount_gt_pivot3 = partition_vec(curr_vec3, pivot_vec, sv, bv);
    int amount_gt_pivot4 = partition_vec(curr_vec4, pivot_vec, sv, bv);
    int amount_gt_pivot5 = partition_vec(curr_vec5, pivot_vec, sv, bv);
    int amount_gt_pivot6 = partition_vec(curr_vec6, pivot_vec, sv, bv);
    int amount_gt_pivot7 = partition_vec(curr_vec7, pivot_vec, sv, bv);
    int amount_gt_pivot8 = partition_vec(curr_vec8, pivot_vec, sv, bv);

    STORE_VEC(arr + l_store, curr_vec); l_store += (8 - amount_gt_pivot);
    STORE_VEC(arr + l_store, curr_vec2); l_store += (8 - amount_gt_pivot2);
    STORE_VEC(arr + l_store, curr_vec3); l_store += (8 - amount_gt_pivot3);
    STORE_VEC(arr + l_store, curr_vec4); l_store += (8 - amount_gt_pivot4);
    STORE_VEC(arr + l_store, curr_vec5); l_store += (8 - amount_gt_pivot5);
    STORE_VEC(arr + l_store, curr_vec6); l_store += (8 - amount_gt_pivot6);
    STORE_VEC(arr + l_store, curr_vec7); l_store += (8 - amount_gt_pivot7);
    STORE_VEC(arr + l_store, curr_vec8); l_store += (8 - amount_gt_pivot8);

    STORE_VEC(arr + r_store + 56, curr_vec); r_store -= amount_gt_pivot;
    STORE_VEC(arr + r_store + 56, curr_vec2); r_store -= amount_gt_pivot2;
    STORE_VEC(arr + r_store + 56, curr_vec3); r_store -= amount_gt_pivot3;
    STORE_VEC(arr + r_store + 56, curr_vec4); r_store -= amount_gt_pivot4;
    STORE_VEC(arr + r_store + 56, curr_vec5); r_store -= amount_gt_pivot5;
    STORE_VEC(arr + r_store + 56, curr_vec6); r_store -= amount_gt_pivot6;
    STORE_VEC(arr + r_store + 56, curr_vec7); r_store -= amount_gt_pivot7;
    STORE_VEC(arr + r_store + 56, curr_vec8); r_store -= amount_gt_pivot8; }

  /* 8 Vektoren die von der linken Seite des Arrays stammen partitionieren und speichern */
  int amount_gt_pivot = partition_vec(vec_left, pivot_vec, sv, bv);
  int amount_gt_pivot2 = partition_vec(vec_left2, pivot_vec, sv, bv);
  int amount_gt_pivot3 = partition_vec(vec_left3, pivot_vec, sv, bv);
  int amount_gt_pivot4 = partition_vec(vec_left4, pivot_vec, sv, bv);
  int amount_gt_pivot5 = partition_vec(vec_left5, pivot_vec, sv, bv);
  int amount_gt_pivot6 = partition_vec(vec_left6, pivot_vec, sv, bv);
  int amount_gt_pivot7 = partition_vec(vec_left7, pivot_vec, sv, bv);
  int amount_gt_pivot8 = partition_vec(vec_left8, pivot_vec, sv, bv);

  STORE_VEC(arr + l_store, vec_left); l_store += (8 - amount_gt_pivot);
  STORE_VEC(arr + l_store, vec_left2); l_store += (8 - amount_gt_pivot2);
  STORE_VEC(arr + l_store, vec_left3); l_store += (8 - amount_gt_pivot3);
  STORE_VEC(arr + l_store, vec_left4); l_store += (8 - amount_gt_pivot4);
  STORE_VEC(arr + l_store, vec_left5); l_store += (8 - amount_gt_pivot5);
  STORE_VEC(arr + l_store, vec_left6); l_store += (8 - amount_gt_pivot6);
  STORE_VEC(arr + l_store, vec_left7); l_store += (8 - amount_gt_pivot7);
  STORE_VEC(arr + l_store, vec_left8); l_store += (8 - amount_gt_pivot8);

  STORE_VEC(arr + r_store + 56, vec_left); r_store -= amount_gt_pivot;
  STORE_VEC(arr + r_store + 56, vec_left2); r_store -= amount_gt_pivot2;
  STORE_VEC(arr + r_store + 56, vec_left3); r_store -= amount_gt_pivot3;
  STORE_VEC(arr + r_store + 56, vec_left4); r_store -= amount_gt_pivot4;
  STORE_VEC(arr + r_store + 56, vec_left5); r_store -= amount_gt_pivot5;
  STORE_VEC(arr + r_store + 56, vec_left6); r_store -= amount_gt_pivot6;
  STORE_VEC(arr + r_store + 56, vec_left7); r_store -= amount_gt_pivot7;
  STORE_VEC(arr + r_store + 56, vec_left8); r_store -= amount_gt_pivot8;

  /* 8 Vektoren die von der rechten Seite des Arrays stammen partitionieren und speichern */
  amount_gt_pivot = partition_vec(vec_right, pivot_vec, sv, bv);
  amount_gt_pivot2 = partition_vec(vec_right2, pivot_vec, sv, bv);
  amount_gt_pivot3 = partition_vec(vec_right3, pivot_vec, sv, bv);
  amount_gt_pivot4 = partition_vec(vec_right4, pivot_vec, sv, bv);
  amount_gt_pivot5 = partition_vec(vec_right5, pivot_vec, sv, bv);
  amount_gt_pivot6 = partition_vec(vec_right6, pivot_vec, sv, bv);
  amount_gt_pivot7 = partition_vec(vec_right7, pivot_vec, sv, bv);
  amount_gt_pivot8 = partition_vec(vec_right8, pivot_vec, sv, bv);

  STORE_VEC(arr + l_store, vec_right); l_store += (8 - amount_gt_pivot);
  STORE_VEC(arr + l_store, vec_right2); l_store += (8 - amount_gt_pivot2);
  STORE_VEC(arr + l_store, vec_right3); l_store += (8 - amount_gt_pivot3);
  STORE_VEC(arr + l_store, vec_right4); l_store += (8 - amount_gt_pivot4);
  STORE_VEC(arr + l_store, vec_right5); l_store += (8 - amount_gt_pivot5);
  STORE_VEC(arr + l_store, vec_right6); l_store += (8 - amount_gt_pivot6);
  STORE_VEC(arr + l_store, vec_right7); l_store += (8 - amount_gt_pivot7);
  STORE_VEC(arr + l_store, vec_right8); l_store += (8 - amount_gt_pivot8);

  STORE_VEC(arr + r_store + 56, vec_right); r_store -= amount_gt_pivot;
  STORE_VEC(arr + r_store + 56, vec_right2); r_store -= amount_gt_pivot2;
  STORE_VEC(arr + r_store + 56, vec_right3); r_store -= amount_gt_pivot3;
  STORE_VEC(arr + r_store + 56, vec_right4); r_store -= amount_gt_pivot4;
  STORE_VEC(arr + r_store + 56, vec_right5); r_store -= amount_gt_pivot5;
  STORE_VEC(arr + r_store + 56, vec_right6); r_store -= amount_gt_pivot6;
  STORE_VEC(arr + r_store + 56, vec_right7); r_store -= amount_gt_pivot7;
  STORE_VEC(arr + r_store + 56, vec_right8); r_store -= amount_gt_pivot8;

  smallest = calc_min(sv); biggest = calc_max(bv);
  return l_store; }

/***
 * vektorisierte Pivot-Strategie*/

/* vektorisierter Zufallszahlengenerator Xoroshiro128+ */
#define VROTL(x, k) /* jedes uint64_t im Vektor rotieren */               \
  _mm256_or_si256(_mm256_slli_epi64((x),(k)),_mm256_srli_epi64((x),64-(k)))

inline __m256i vnext(__m256i &s0, __m256i &s1) {
  s1 = _mm256_xor_si256(s0, s1);      /* Vektoren s1 und s0 modifizieren */
  s0 = _mm256_xor_si256(_mm256_xor_si256(VROTL(s0, 24), s1),
                        _mm256_slli_epi64(s1, 16));
  s1 = VROTL(s1, 37);
  return _mm256_add_epi64(s0, s1); }        /* Zufallsvektor zurückgeben */

/* ZZ auf den Bereich zwischen 0 und bound - 1 transformieren */
inline __m256i rnd_epu32(__m256i rnd_vec, __m256i bound) {
  __m256i even = _mm256_srli_epi64(_mm256_mul_epu32(rnd_vec, bound), 32);
  __m256i odd = _mm256_mul_epu32(_mm256_srli_epi64(rnd_vec, 32), bound);
  return _mm256_blend_epi32(odd, even, 0b01010101); }


/* Durchschnitt aus zwei Integern ohne Überlauf
 * http://aggregate.org/MAGIC/#Average%20of%20Integers */
inline int average(int a, int b) { return (a & b) + ((a ^ b) >> 1); }

inline int get_pivot(int *arr, const int left, const int right){
  auto bound = _mm256_set1_epi32(right - left + 1);
  auto left_vec = _mm256_set1_epi32(left);

  /* Seeds für vektorisierten Zufallszahlengenerator */
  auto s0 = _mm256_setr_epi64x(8265987198341093849, 3762817312854612374,
                               1324281658759788278, 6214952190349879213);
  auto s1 = _mm256_setr_epi64x(2874178529384792648, 1257248936691237653,
                               7874578921548791257, 1998265912745817298);
  s0 = _mm256_add_epi64(s0, _mm256_set1_epi64x(left));
  s1 = _mm256_sub_epi64(s1, _mm256_set1_epi64x(right));

  __m256i v[9];             /* ausgedünntes Array für Median der Mediane */
  for (int i = 0; i < 9; ++i) {           /* ausgedünntes Array befüllen */
    auto result = vnext(s0, s1);     /* Vektor mit 4 zufälligen uint64_t */
    result = rnd_epu32(result, bound);    /* ZZ zwischen 0 und bound - 1 */
    result = _mm256_add_epi32(result, left_vec);      /* Indices für arr */
    v[i] = _mm256_i32gather_epi32(arr, result, sizeof(uint32_t)); }

  /* Median-Netzwerk für 9 Elemente */
  COEX_VERTICAL(v[0], v[1]); COEX_VERTICAL(v[2], v[3]);     /* Schritt 1 */
  COEX_VERTICAL(v[4], v[5]); COEX_VERTICAL(v[6], v[7]);
  COEX_VERTICAL(v[0], v[2]); COEX_VERTICAL(v[1], v[3]);     /* Schritt 2 */
  COEX_VERTICAL(v[4], v[6]); COEX_VERTICAL(v[5], v[7]);
  COEX_VERTICAL(v[0], v[4]); COEX_VERTICAL(v[1], v[2]);     /* Schritt 3 */
  COEX_VERTICAL(v[5], v[6]); COEX_VERTICAL(v[3], v[7]);
  COEX_VERTICAL(v[1], v[5]); COEX_VERTICAL(v[2], v[6]);     /* Schritt 4 */
  COEX_VERTICAL(v[3], v[5]); COEX_VERTICAL(v[2], v[4]);     /* Schritt 5 */
  COEX_VERTICAL(v[3], v[4]);                                /* Schritt 6 */
  COEX_VERTICAL(v[3], v[8]);                                /* Schritt 7 */
  COEX_VERTICAL(v[4], v[8]);                                /* Schritt 8 */

  SORT_8(v[4]);                    /* die acht Mediane in v[4] sortieren */
  return average(_mm256_extract_epi32(v[4], 3),            /* Pivot-Wert */
                 _mm256_extract_epi32(v[4], 4)); }

inline void qs_core(int *arr, int left, int right,
                    bool choose_avg = false, const int avg = 0) {
  if (right - left < 513) { /* Sortiernetzwerke für kleine Teilarrays */
    /*** bei Multithreading dynamischer ausgerichteter Buffer
     * std::vector<int> v(600);
     * int* buffer = v.data() + (reinterpret_cast<std::uintptr_t>
     *                          (v.data()) % 32) / sizeof(int);
     * oder bei wenigen threads auf den Stack
     * __m256i buffer[66]; */
    static __m256i buffer[66]; /* Buffer für Sortiernetzwerke (singlethreaded) */
    int* buff = reinterpret_cast<int *>(buffer);
    sort_int_sorting_network(arr + left, buff, right - left + 1);
    return; }
  /* avg ist Durchschnitt von größten und kleinsten Wert im Array */
  int pivot = choose_avg ? avg : get_pivot(arr, left, right);
  int smallest = INT32_MAX;     /* kleinster Wert nach Partitionierung */
  int biggest = INT32_MIN;        /* größter Wert nach Partitionierung */
  int bound = partition_vectorized_64(arr, left, right + 1, pivot, smallest, biggest);
  /* Anteil der kleinereren Partition am Array */
  double ratio = (std::min(right-(bound-1),bound-left)/double(right-left+1));
  /* falls unbalancierte Teilarrays, Pivot-Berechnung ändern */
  if (ratio < 0.2) { choose_avg = !choose_avg; }
  if (pivot != smallest)      /* Werte im linken Teilarray verschieden */
    qs_core(arr, left, bound - 1, choose_avg, average(smallest, pivot));
  if (pivot + 1 != biggest)  /* Werte im rechten Teilarray verschieden */
    qs_core(arr, bound, right, choose_avg, average(biggest, pivot)); }

/* Rekursion für Quickselect */
inline void qsel_core(int *arr, int left, int right, int k,
                      bool choose_avg = false, const int avg = 0) {
  if (right - left < 256) {
    /* für wenige Elemente C++'s nth_element benutzen */
    std::nth_element(arr + left, arr + k, arr + right + 1);
    return; }
  /* avg ist Durchschnitt von größten und kleinsten Wert im Array */
  int pivot = choose_avg ? avg : get_pivot(arr, left, right);
  int smallest = INT32_MAX;     /* kleinster Wert nach Partitionierung */
  int biggest = INT32_MIN;        /* größter Wert nach Partitionierung */
  int bound = partition_vectorized_64(arr, left, right + 1, pivot,
                                                 smallest, biggest);
  /* Anteil der kleinereren Partition am Array */
  double ratio = (std::min(right-(bound-1),bound-left)/double(right-left+1));
  /* falls unbalancierte Teilarrays, Pivot-Berechnung ändern */
  if (ratio < 0.2) { choose_avg = !choose_avg; }
  if(k < bound){ /* k ist auf der linken Seite von bound */
    if(pivot != smallest) /* Werte im linken Teilarray verschieden */
      qsel_core(arr, left, bound-1, k, choose_avg, average(smallest, pivot));
  } else { /* k ist auf der rechten Seite von bound */
    if(pivot + 1 != biggest) /* Werte im rechten Teilarray verschieden */
      qsel_core(arr, bound, right, k, choose_avg, average(biggest, pivot));
  }}

inline void qs_core_omp(int *arr, int left, int right,
                    bool choose_avg = false, const int avg = 0) {
  if (right - left < 513) { /* Sortiernetzwerke für kleine Teilarrays */
    /*** bei vielen Threads
     * std::vector<int> v(600);
     * int* buffer = v.data() + (reinterpret_cast<std::uintptr_t>
     *                                  (v.data()) % 32) / sizeof(int); */
    __m256i buffer[66]; /* Buffer bei wenigen Threads */
    int* buff = reinterpret_cast<int *>(buffer);
    sort_int_sorting_network(arr + left, buff, right - left + 1);
    return; }
  /* avg ist Durchschnitt von größten und kleinsten Wert im Array */
  int pivot = choose_avg ? avg : get_pivot(arr, left, right);
  int smallest = INT32_MAX;     /* kleinster Wert nach Partitionierung */
  int biggest = INT32_MIN;        /* größter Wert nach Partitionierung */
  int bound = partition_vectorized_64(arr, left, right + 1, pivot, smallest, biggest);
  /* Anteil der kleinereren Partition am Array */
  double ratio = (std::min(right-(bound-1),bound-left)/double(right-left+1));
  /* falls unbalancierte Teilarrays, Pivot-Berechnung ändern */
  if (ratio < 0.2) { choose_avg = !choose_avg; }
  if (pivot != smallest) {    /* Werte im linken Teilarray verschieden */
#pragma omp task final(bound - left < 50000) mergeable
    qs_core_omp(arr, left, bound - 1, choose_avg, average(smallest, pivot)); }
  if (pivot + 1 != biggest){ /* Werte im rechten Teilarray verschieden */
    qs_core_omp(arr, bound, right, choose_avg, average(biggest, pivot)); }}

}/* namespace __internal Ende */

/* diese Funktion zur Bestimmung des k-größten Elements aufrufen */
inline void quickselect(int *arr, int n, int k) {
  __internal::qsel_core(arr, 0, n - 1, k); }

/* diese Funktion zum sortieren aufrufen */
inline void quicksort(int *arr, int n) { __internal::qs_core(arr, 0, n - 1); }

/* diese Funktion zum parallelen sortieren aufrufen */
inline void quicksort_omp(int *arr, int n) {
#pragma omp parallel default(none) shared(arr, n)
#pragma omp single nowait
  {__internal::qs_core_omp(arr, 0, n - 1); }}

} /* namespace avx2 Ende */

#endif /* AVX2SORT_H */
