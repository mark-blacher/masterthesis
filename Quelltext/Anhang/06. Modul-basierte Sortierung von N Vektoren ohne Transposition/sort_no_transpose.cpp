#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <iostream>

#define COEX_VERTICAL(a, b){  /* berechne acht Vergleichsmodule */ 	     \
  __m256i c = a; a = _mm256_min_epi32(a, b); b = _mm256_max_epi32(c, b);}

/* Blend-Maske aus Permutations-Indices für aufsteigendes Sortieren */
#define ASC(a, b, c, d, e, f, g, h)                                       \
  ((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) |     \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0)

/* vektorisiertes compare-exchange mit Permutation */
#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){                  \
  __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);       \
  __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask);      \
  __m256i min = _mm256_min_epi32(permuted, vec);                          \
  __m256i max = _mm256_max_epi32(permuted, vec);                          \
  vec = _mm256_blend_epi32(min, max, MASK(a, b, c, d, e, f, g, h));}

/* vektorisiertes compare-exchange mit Shuffle */
#define COEX_SHUFFLE(vec, a, b, c, d, e, f, g, h, MASK){                  \
  constexpr auto shuffle_mask = _MM_SHUFFLE(d, c, b, a);                  \
  __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);             \
  __m256i min = _mm256_min_epi32(shuffled, vec);                          \
  __m256i max = _mm256_max_epi32(shuffled, vec);                          \
  vec = _mm256_blend_epi32(min, max, MASK(a, b, c, d, e, f, g, h));}

#define REVERSE_VEC(v) /* kehre Reihenfolge der Zahlen im Vektor um */	  \
  v = _mm256_permutevar8x32_epi32(v,                                      \
   _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));

#define CV(a, b) COEX_VERTICAL(a, b) /* Alias für weniger Code */

/* sortiere in 8 Spalten jeweils 16 int mit 60 Modulen */
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

/* Hilfsfunktion zum spaltenweisen sortieren und mergen */
inline void oddeven_mergesort(__m256i *vecs, int N, const int s = 2) {
  for (int t = s; t < N * 2; t *= 2) {
    for (int l = 0; l < N; l += t) {
      const int bound = std::min(l + t / 2, N - t / 2);
      for (int i = l; i < bound; ++i) {
        COEX_VERTICAL(vecs[i], vecs[i + t / 2]); }
      for (int j = t / 4; j > 0; j /= 2) {
        for (int i = j; i < t - j * 2; i += 2 * j) {
          const int bound = std::min(i + l + j, N - j);
          for (int k = i + l; k < bound; ++k) {
            COEX_VERTICAL(vecs[k], vecs[k + j]); }}}}}}

/* Hilfsfunktion zur Ermittlung der nächsten Zweierpotenz von x */
inline int next_power_of_two(int x) {
  x--; x |= x >> 1; x |= x >> 2; x |= x >> 4;
  x |= x >> 8; x |= x >> 16; x++;
  return x;}

/* N Vektoren sortieren, d. h. N * 8 int */
void no_transpose_sort(__m256i *vecs, const int N) {
  /* Phase 1: Sortiere Spalten */
  /* spaltenweise Sechzehnergruppen mit 60 Modulnetzwerk sortieren */
  for (int i = 0; i < N - N % 16; i += 16){sort_16_int_vertical(vecs + i);}
  /* sortiere letzte Vektoren mit Odd-Even Mergesort, falls N % 16 != 0 */
  oddeven_mergesort(vecs + N - N % 16, N % 16);
  /* Merge in Spalten Sechzehnergruppen */
  oddeven_mergesort(vecs, N, 16);

  /* Phase 2: Mit Bitonic Merge die sortierten Spalten mergen */
  const int npot = next_power_of_two(N);
  for (int m = npot/2; m > 0; m/=2) {
    for (int i = m; i < N; i += 2 * m) {
      const int bound = std::min(N, i + m);
      for (int j = i, k=1; j < bound; ++j, ++k) {
        vecs[j] = _mm256_shuffle_epi32(vecs[j], _MM_SHUFFLE(2,3,0,1));
        COEX_VERTICAL(vecs[i - k], vecs[j])}}}
  for (int l = 0; l < N; ++l) {
    COEX_SHUFFLE(vecs[l], 1, 0, 3, 2, 5, 4, 7, 6, ASC);}
  for (int m = npot / 2; m > 0; m/=2) {
    for (int i = m; i < N; i += 2 * m) {
      const int bound = std::min(N, i + m);
      for (int j = i, k=1; j < bound; ++j, ++k) {
        vecs[j] = _mm256_shuffle_epi32(vecs[j], _MM_SHUFFLE(0,1,2,3));
        COEX_VERTICAL(vecs[i - k], vecs[j])}}}
  for (int l = 0; l < N; ++l) {
    COEX_SHUFFLE(vecs[l], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE(vecs[l], 1, 0, 3, 2, 5, 4, 7, 6, ASC);}
  for (int m = npot / 2; m > 0; m /= 2) {
    for (int i = m; i < N; i += 2 * m) {
      const int bound = std::min(N, i + m);
      for (int j = i, k = 1; j < bound; ++j, ++k) {
        REVERSE_VEC(vecs[j]);
        COEX_VERTICAL(vecs[i - k], vecs[j])}}}
  for (int l = 0; l < N; ++l) {
    COEX_PERMUTE(vecs[l], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE(vecs[l], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE(vecs[l], 1, 0, 3, 2, 5, 4, 7, 6, ASC);}}

int main(){
  /* teste bis 1024 Vektoren ob richtig sortiert */
  for (int N = 0; N < 1025; ++N) {
    __m256i vecs[N];
    /* befülle Vektoren mit Zufallszahlen */
    auto *arr = reinterpret_cast<int *>(vecs);
    for (int i = 0; i < N * 8; ++i) {
      arr[i] = rand();
    }
    std::vector<int> std_vec(arr, arr + N * 8); /* kopiere vor Sortierung */
    no_transpose_sort(vecs, N); /* sortiere N * 8 int aufsteigend */

    std::sort(begin(std_vec), end(std_vec)); /* sortiere mit std */
    std::vector<int> result(arr, arr + N * 8); /* kopiere array */

    if (std_vec != result) {
      std::cout << "\nfail! nicht sortiert!!! " << N;
      exit(0);
    }}
  std::cout << "alles ok";}

