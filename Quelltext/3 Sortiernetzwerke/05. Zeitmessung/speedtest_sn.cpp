#include <cmath>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>

using namespace std;
using namespace std::chrono;

/*
 * das Programm testet drei verschiedene Vektorisierungsstrategien
 * für Sortiernetzwerke
 */

#define LOAD_VEC(arr) _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr))
#define STORE_VEC(arr, vec)                                                   \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr), vec)

#define COEX_VERTICAL(a, b){                                                   \
    auto vec_tmp = a;                                                          \
    a = _mm256_min_epi32(a, b);                                                \
    b = _mm256_max_epi32(vec_tmp, b);}


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
  v2 = _mm256_blend_epi32(b1, v2, 0b10101010);
}

#define ASC(a, b, c, d, e, f, g, h)                                             \
  (((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) |          \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0))

#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){                       \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);          \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask);         \
    __m256i min = _mm256_min_epi32(permuted, vec);                             \
    __m256i max = _mm256_max_epi32(permuted, vec);                             \
    constexpr uint8_t blend_mask = MASK(a, b, c, d, e, f, g, h);         \
    vec = _mm256_blend_epi32(min, max, blend_mask);}


#define COEX_SHUFFLE(vec, a, b, c, d, e, f, g, h, MASK){                       \
    constexpr uint8_t shuffle_mask = _MM_SHUFFLE(d, c, b, a);                  \
    __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);                \
    __m256i min = _mm256_min_epi32(shuffled, vec);                             \
    __m256i max = _mm256_max_epi32(shuffled, vec);                             \
    constexpr uint8_t blend_mask = MASK(a, b, c, d, e, f, g, h);               \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

#define MIN(x, y) ((x)+((((y)-(x))>>31)&((y)-(x)))) /* min von 2 int */
#define MAX(x, y) ((x)-((((x)-(y))>>31)&((x)-(y)))) /* max von 2 int */

#define REVERSE_VEC(vec){                                                      \
    vec = _mm256_permutevar8x32_epi32(                                         \
        vec, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));}

/* N Vektoren mergen, N ist Anzahl der Vektoren, N % 2 == 0 und N > 0
 * s = 2 bedeuted, dass jeweils zwei Vektoren bereits sortiert sind */
inline void bitonic_merge_16(__m256i *vecs, const int N, const int s = 2) {
  for (int t = s * 2; t < 2 * N; t *= 2) {
    for (int l = 0; l < N; l += t) {
      for (int j = MAX(l + t - N, 0); j < t/2 ; j += 2) {
        REVERSE_VEC(vecs[l + t - 1 - j]);
        REVERSE_VEC(vecs[l + t - 2 - j]);
        COEX_VERTICAL(vecs[l + j], vecs[l + t - 1 - j]);
        COEX_VERTICAL(vecs[l + j + 1], vecs[l + t - 2 - j]); }}
    for (int m = t / 2; m > 4; m /= 2) {
      for (int k = 0; k < N - m / 2; k += m) {
        const int bound = MIN((k + m / 2), N - (m / 2));
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

/* diese Funktion benutzen falls Gruppen von 16 Vektoren sortiert vorliegen */
inline void bitonic_merge_128(__m256i *vecs, const int N, const int s = 16) {
  const int remainder16 = N - N % 16;
  const int remainder8 = N - N % 8;
  for (int t = s * 2; t < 2 * N; t *= 2) {
    for (int l = 0; l < N; l += t) {
      for (int j = MAX(l + t - N, 0); j < t/2 ; j += 2) {
        REVERSE_VEC(vecs[l + t - 1 - j]);
        REVERSE_VEC(vecs[l + t - 2 - j]);
        COEX_VERTICAL(vecs[l + j], vecs[l + t - 1 - j]);
        COEX_VERTICAL(vecs[l + j + 1], vecs[l + t - 2 - j]); }}
    for (int m = t / 2; m > 16; m /= 2) {
      for (int k = 0; k < N - m / 2; k += m) {
        const int bound = MIN((k + m / 2), N - (m / 2));
        for (int j = k; j < bound; j += 2) {
          COEX_VERTICAL(vecs[j], vecs[m / 2 + j]);
          COEX_VERTICAL(vecs[j + 1], vecs[m / 2 + j + 1]); }}}
    for (int j = 0; j < remainder16; j += 16) { /* entrollt */
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

/* zum mergen der acht sortierten Spalten mit jeweils 16 Elementen */
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

/* arr ist das Array, dass sortiert wird
 * n ist Anzahl der Elemente im Array
 * n  Modulo 16 muss 0 sein und n größer 0
 * buffer ist ausgerichteter Speicher mit genügend Platz für arr */
inline void sort_bitonic_16(int *arr, __m256i* buffer, const int n) {
  for (int i = 0; i < n / 8; i += 2) {
    buffer[i] = LOAD_VEC(arr + i * 8);
    buffer[i + 1] = LOAD_VEC(arr + i * 8 + 8);
    sort_16(buffer[i], buffer[i + 1]);
  }
  bitonic_merge_16(buffer, n / 8, 2);
  for (int i = 0; i < n / 8; i += 2) {
    STORE_VEC(arr + i * 8, buffer[i]);
    STORE_VEC(arr + i * 8 + 8, buffer[i + 1]);
  }
}

/* schnellste der drei Strategien */
inline void sort_bitonic_best(int *arr, __m256i* buffer, const int n) {
  const int N = n/8;
  for (int j = 0; j < N - N % 16; j+=16) {
    for (int i = j; i < 16 + j; ++i) {
      buffer[i] = LOAD_VEC(arr + i * 8);
    }
    sort_16_int_vertical(buffer + j);
    merge_8_columns_with_16_elements(buffer + j);
  }
  for (int i = N - N % 16; i < N; i += 2) {
    buffer[i] = LOAD_VEC(arr + i * 8);
    buffer[i + 1] = LOAD_VEC(arr + i * 8 + 8);
    sort_16(buffer[i], buffer[i + 1]);
  }
  bitonic_merge_16(buffer + N - N % 16, N % 16, 2);
  bitonic_merge_128(buffer, N, 16);
  for (int i = 0; i < N; i += 2) {
    STORE_VEC(arr + i * 8, buffer[i]);
    STORE_VEC(arr + i * 8 + 8, buffer[i + 1]);
  }
}

/* so einfach wie möglich (wie im Hauptkapitel beschrieben) */
#define SORT_8_NAIVE(vec){ /* sortiere aufsteigend 8 int */ \
COEX_PERMUTE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC); /* Schritt 1 */ \
COEX_PERMUTE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC); /* Schritt 2 */ \
COEX_PERMUTE(vec, 0, 2, 1, 3, 4, 6, 5, 7, ASC); /* Schritt 3 */ \
COEX_PERMUTE(vec, 7, 6, 5, 4, 3, 2, 1, 0, ASC); /* Schritt 4 */ \
COEX_PERMUTE(vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC); /* Schritt 5 */ \
COEX_PERMUTE(vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC);} /* Schritt 6 */

/* N Vektoren naive sortieren, d. h. insgesamt N * 8 int */
inline void sort_bitonic(__m256i *vecs, const int N){
  for (int i = 0; i < N; ++i) SORT_8_NAIVE(vecs[i]); /* sortiere Achtergruppen */
  for (int t = 2; t < N * 2; t *= 2) { /* merge 2 dann 4, 8 ... Vektoren */
    for (int l = 0; l < N; l += t){
      for (int j = std::max(l + t - N, 0); j < t/2 ; j += 1) {
        REVERSE_VEC(vecs[l + t - 1 - j]);
        COEX_VERTICAL(vecs[l + j], vecs[l + t - 1 - j]);}}
    for (int m = t / 2; m > 1; m /= 2){
      for (int k = 0; k < N - m / 2; k += m) {
        const int bound = std::min((k + m / 2), N - (m / 2));
        for (int j = k; j < bound; j += 1){
          COEX_VERTICAL(vecs[j], vecs[m / 2 + j]);}}}
    for (int i = 0; i < N; i += 1){
      COEX_PERMUTE(vecs[i], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
      COEX_PERMUTE(vecs[i], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
      COEX_PERMUTE(vecs[i], 1, 0, 3, 2, 5, 4, 7, 6, ASC); }}}

void sort_bitonic_naive(int* arr, __m256i* buffer, const uint32_t n){
  for (uint32_t i = 0; i < n/8; i += 2) {
    buffer[i] = LOAD_VEC(arr + i * 8);
    buffer[i + 1] = LOAD_VEC(arr + i * 8 + 8);
  }
  sort_bitonic(buffer, n/8);
  for (uint32_t i = 0; i < n/8; i += 1) {
    STORE_VEC(arr + i * 8, buffer[i]);
  }
}

/* Zeit messen */

__m256i buffer[1000000];

/* Korrektness von schnellster Sortierung testen */
void test_correctness(){
  for (int k = 16; k < 16000; k+=16) {
    volatile int n = k;
    vector<int> v(k);
    auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
    for (int i = 0; i < n; ++i) {
      v[i] = die();
    }
    auto a = v;
    auto b = v;
    sort(begin(a), end(a));
    sort_bitonic_best(b.data(), buffer, k);
    if(a != b){
      cerr << "fail " << n << endl;
      exit(-1);
    }
  }
  cout << "Best sortiert korrekt!" << endl;
}
int main() {
  test_correctness();
  bool calculate_speedups_for_diagram = false; /* auf true setzen falls Zeitmessung für Diagramm */
  if(calculate_speedups_for_diagram){
    volatile int n = 0;

    vector<int> vec_n(26);
    vector<int> vec_runs(26);
    vector<double> vec_speedups(26 * 3, 0);
    for (int k = 16, i=0; k < 16384; k+=640, ++i) vec_n[i] = k;
    for (int i = 0; i < 26; ++i) vec_runs[i] = int(1280000000.0 / (log2(vec_n[i]) * vec_n[i]));

    int repeating = 10;
    for (int s = 0; s < repeating; ++s) {
      for (int i = 0; i < 26; ++i) {
        n = vec_n[i];
        int runs = vec_runs[i];

        vector<int> v(n);
        auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
        for (int r = 0; r < n; ++r) {
          v[r] = die();
        }

        auto a = v;
        auto b = v;
        auto c = v;
        auto d = v;

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
          auto tic = system_clock::now();
          for (int j = 0; j < runs; ++j) {
            memcpy(a.data(), v.data(), sizeof(int) * n);
            sort(begin(a), end(a));
          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          dur_std = dur;
        }
        {
          /* step based naive */
          auto tic = system_clock::now();
          for (int j = 0; j < runs; ++j) {
            memcpy(b.data(), v.data(), sizeof(int) * n);
            sort_bitonic_naive(b.data(), buffer, n);
          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          vec_speedups[i] += dur > 0.000001 ? dur_std/double(dur) : 0;
          if(a != b){cerr << "step based naive failed!!!"; exit(-1);} ;
        }

        {
          /* step based optimized */;
          auto tic = system_clock::now();
          for (int j = 0; j < runs; ++j) {
            memcpy(c.data(), v.data(), sizeof(int) * n);
            sort_bitonic_16(c.data(), buffer, n);
          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          vec_speedups[i + 26] += dur > 0.000001 ? dur_std/double(dur) : 0;
          if(a != c){cerr << "step based optimized failed!!!"; exit(-1);} ;
        }

        {
          /* best */;
          auto tic = system_clock::now();
          for (int j = 0; j < runs; ++j) {
            memcpy(d.data(), v.data(), sizeof(int) * n);
            sort_bitonic_best(d.data(), buffer, n);
          }
          auto toc = system_clock::now();
          auto dur = duration<double>(toc - tic).count() - copy_time;
          vec_speedups[i + 52] += dur > 0.00001 ? dur_std / double(dur) : 0;
          if(a != d){cerr << "best failed!!!"; exit(-1);} ;
        }
        cout << "n: " << n << " in repeating " << s << " finished" << endl;
      }
      cout << endl;
    }
    cout << "data_from_cpp = c(";
    for (int k = 0; k < 26 * 3 - 1; ++k) {
      cout << (vec_speedups[k] / repeating) << ", ";
    }
    cout << (vec_speedups[26 * 3 - 1] / repeating) << ")\n";
  }else {

    volatile int n = 2000;
    const int runs = 100000; // repetitions of calculation
    vector<int> v(n);

    auto die = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
    for (int i = 0; i < n; ++i) {
      v[i] = die();
    }

    auto a = v;
    auto b = v;
    auto c = v;
    auto d = v;

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
      cout << "step based naive:";
      auto tic = system_clock::now();
      for (int j = 0; j < runs; ++j) {
        memcpy(b.data(), v.data(), sizeof(int) * n);
        sort_bitonic_naive(b.data(), buffer, n);
      }
      auto toc = system_clock::now();
      auto dur = duration<double>(toc - tic).count() - copy_time;
      cout << "\n" << dur << " sec";
      if (dur_std > 0.001) cout << "\nspeed-up: " << dur_std / double(dur);
      cout << boolalpha << "\nis_sorted: " << is_sorted(begin(b), end(b)) << endl;
      cout << "same vectors: " << (a == b) << endl << endl;
    }

    {
      cout << "step based optimized:";
      auto tic = system_clock::now();
      for (int j = 0; j < runs; ++j) {
        memcpy(c.data(), v.data(), sizeof(int) * n);
        sort_bitonic_16(c.data(), buffer, n);
      }
      auto toc = system_clock::now();
      auto dur = duration<double>(toc - tic).count() - copy_time;
      cout << "\n" << dur << " sec";
      if (dur_std > 0.001) cout << "\nspeed-up: " << dur_std / double(dur);
      cout << boolalpha << "\nis_sorted: " << is_sorted(begin(c), end(c)) << endl;
      cout << "same vectors: " << (a == c) << endl << endl;
    }

    {
      cout << "best:";
      auto tic = system_clock::now();
      for (int j = 0; j < runs; ++j) {
        memcpy(d.data(), v.data(), sizeof(int) * n);
        sort_bitonic_best(d.data(), buffer, n);
      }
      auto toc = system_clock::now();
      auto dur = duration<double>(toc - tic).count() - copy_time;
      cout << "\n" << dur << " sec";
      if (dur_std > 0.001) cout << "\nspeed-up: " << dur_std / double(dur);
      cout << boolalpha << "\nis_sorted: " << is_sorted(begin(c), end(c)) << endl;
      cout << "same vectors: " << (a == d) << endl << endl;
    }
  }
}