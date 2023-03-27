#include <cstdio>
#include <immintrin.h>

/*
 * dieses Codebeispiel demonstriert vektorisierten Bitonic Mergesort
 * für 16 int
 * es wurden keine Optimierungen vorgenommen
 */

/* lade aus Array 8 int ins Vektorregister */
#define LOAD_VEC(arr) _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr))

/* speichere Vektor ins Array */
#define STORE_VEC(arr, vec)                                               \
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr), vec)

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

#define REVERSE_VEC(v){ /* kehre Reihenfolge der Zahlen im Vektor um */   \
  v = _mm256_permutevar8x32_epi32(v,                                      \
          _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));}

#define COEX_VERTICAL(a, b){ /* berechne acht Vergleichsmodule */ 	     \
  __m256i c = a; a = _mm256_min_epi32(a, b); b = _mm256_max_epi32(c, b);}

#define LAST_3_STEPS(v){ /* letzten drei Schritte des Netzwerks */        \
  COEX_PERMUTE(v, 4, 5, 6, 7, 0, 1, 2, 3, 0b11110000);                    \
  COEX_PERMUTE(v, 2, 3, 0, 1, 6, 7, 4, 5, 0b11001100);                    \
  COEX_PERMUTE(v, 1, 0, 3, 2, 5, 4, 7, 6, 0b10101010);}

#define MERGE_16(v1, v2){ /* merge zwei sortierte Vektoren */             \
  REVERSE_VEC(v2);                                                        \
  COEX_VERTICAL(v1, v2); /* Schritt 7 */                                  \
  LAST_3_STEPS(v1); LAST_3_STEPS(v2);} /* Schritte 8, 9 und 10 */

#define SORT_16(v1, v2){ /* sortiert 16 int */                            \
  SORT_8(v1); SORT_8(v2); /* sortiere Vektoren v1 und v2 */               \
  MERGE_16(v1, v2);} /* merge v1 und v2 */

int main() {
  /* Teste Sortierung von 16 int */
  int arr[16] = {7, 8, 13, 12, 4, 3, 2, 9, 6, 11, 5, 1, 16, 10, 14, 15};
  __m256i vecs[2] = {LOAD_VEC(arr), LOAD_VEC(arr + 8)};
  SORT_16(vecs[0], vecs[1]); /* sortiere zwei Vektoren */
  STORE_VEC(arr, vecs[0]); STORE_VEC(arr + 8, vecs[1]);
  for(int i=0; i < 16; ++i) printf("%d ", arr[i]);}