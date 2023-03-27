#include <cstdio>
#include <stdlib.h>
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <iostream>

/*
 * dieses Codebeispiel demonstriert vektorisierten Bitonic Mergesort
 * n muss ohne Rest durch acht teilbar sein
 * es wurden keine Optimierungen vorgenommen
 */

/* vektorisiertes compare-exchange mit Permutation */
#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){                       \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);          \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask);         \
    __m256i min = _mm256_min_epi32(permuted, vec);                             \
    __m256i max = _mm256_max_epi32(permuted, vec);                             \
    vec = _mm256_blend_epi32(min, max, MASK);}

#define SORT_8(vec){                       /* sortiere aufsteigend 8 int */    \
    COEX_PERMUTE(vec, 1, 0, 3, 2, 5, 4, 7, 6, 0b10101010);  /* Schritt 1 */    \
    COEX_PERMUTE(vec, 2, 3, 0, 1, 6, 7, 4, 5, 0b11001100);  /* Schritt 2 */    \
    COEX_PERMUTE(vec, 0, 2, 1, 3, 4, 6, 5, 7, 0b01000100);  /* Schritt 3 */    \
    COEX_PERMUTE(vec, 7, 6, 5, 4, 3, 2, 1, 0, 0b11110000);  /* Schritt 4 */    \
    COEX_PERMUTE(vec, 2, 3, 0, 1, 6, 7, 4, 5, 0b11001100);  /* Schritt 5 */    \
    COEX_PERMUTE(vec, 1, 0, 3, 2, 5, 4, 7, 6, 0b10101010);} /* Schritt 6 */

#define REVERSE_VEC(v) /* kehre Reihenfolge der Zahlen im Vektor um */	  \
  v = _mm256_permutevar8x32_epi32(v,                                      \
   _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));

#define COEX_VERTICAL(a, b){  /* berechne acht Vergleichsmodule */ 	     \
  __m256i c = a; a = _mm256_min_epi32(a, b); b = _mm256_max_epi32(c, b);}

#define LAST_3_STEPS(v){ /* letzten drei Schritte des Netzwerks */       \
  COEX_PERMUTE(v, 4, 5, 6, 7, 0, 1, 2, 3, 0b11110000);                 \
  COEX_PERMUTE(v, 2, 3, 0, 1, 6, 7, 4, 5, 0b11001100);                 \
  COEX_PERMUTE(v, 1, 0, 3, 2, 5, 4, 7, 6, 0b10101010);}

/* N Vektoren sortieren, d. h. insgesamt  N * 8 int */
inline void sort_bitonic(__m256i *vecs, const int N){
  for (int i = 0; i < N; ++i) SORT_8(vecs[i]); /* sortiere Achtergruppen */
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
    for (int i = 0; i < N; i += 1){ /* COEX_VERTIKAL nicht anwendbar */
      LAST_3_STEPS(vecs[i]);}}}

int main() {
  /* in jeder Iteration werden N * 8 int sortiert */
  for (int N = 0; N < 513; ++N) {
    __m256i vecs[N];

    /* befülle Vektoren mit Zufallszahlen */
    auto *arr = reinterpret_cast<int *>(vecs);
    for (int i = 0; i < N * 8; ++i) {
      arr[i] = rand();
    }

    std::vector<int> std_vec(arr, arr + N * 8); /* kopiere vor Sortierung */
    sort_bitonic(vecs, N); /* sortiere N * 8 int aufsteigend */

    std::sort(begin(std_vec), end(std_vec)); /* sortiere mit std */
    std::vector<int> bitonic_result(arr, arr + N * 8); /* kopiere array */

    /* ist das Ergebnis mit std::sort identisch */
    if (std_vec != bitonic_result) {
        std::cout << "Fail: " << N;
        exit(0);
    }
  }
  std::cout << "Alles OK\n";
}